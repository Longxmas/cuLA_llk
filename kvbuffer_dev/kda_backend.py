from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.utils import is_cpu, is_npu
from sglang.srt.utils.common import rank0_log

# KDA always uses the triton causal_conv1d_fn (no CUDA override).
# Only causal_conv1d_update needs platform-specific overrides for decode.
if is_npu():
    from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_update_npu

    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_update_cpu

    causal_conv1d_update = causal_conv1d_update_cpu

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class KDAKernelDispatcher:
    """Dispatches KDA kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonKDAKernel()

        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            from sglang.srt.layers.attention.linear.kernels.kda_cutedsl import (
                CuteDSLKDAKernel,
            )

            self.decode_kernel = CuteDSLKDAKernel()
        else:
            raise ValueError(
                f"Unsupported KDA decode backend: {decode_backend}. "
                "KDA currently only supports 'triton'."
            )

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cula():
            from sglang.srt.layers.attention.linear.kernels.kda_cula import (
                CulaKDAKernel,
            )

            self.extend_kernel = CulaKDAKernel()
        else:
            raise ValueError(
                f"Unsupported KDA prefill backend: {prefill_backend}. "
                "KDA currently supports 'triton' and 'cula'."
            )

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"KDA kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__} "
            f"packed_decode={self.supports_packed_decode}"
        )

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Attempt packed decode. Returns output tensor or None if the decode
        kernel does not support packed decode."""
        if not self.supports_packed_decode:
            return None
        return self.decode_kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            **kwargs,
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Target verify for speculative decoding."""
        return self.decode_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )


class KDAAttnBackend(MambaAttnBackendBase):
    """Attention backend for KDA (Kimi Delta Attention) linear attention."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0]
            .transpose(-1, -2)
            .shape
        )
        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        self.kernel_dispatcher = KDAKernelDispatcher(decode_backend, prefill_backend)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        if self.forward_metadata.has_mamba_track_mask:
            self.forward_metadata.mamba_track_mask_indices = (
                forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
            )
            if not forward_batch.forward_mode.is_target_verify():
                self.forward_metadata.conv_states_mask_indices = (
                    forward_batch.mamba_track_indices[
                        self.forward_metadata.mamba_track_mask_indices
                    ]
                )

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states.transpose(-1, -2),
            layer.conv_weights,
            layer.bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )

        # Skip split + reshape by consuming the packed mixed_qkv directly in a
        # single fused Triton kernel (KDA per-K gate variant of GDN PR #20627).
        #
        # The packed kernel hard-assumes one token per sequence (T=1): it has no
        # query_start_loc / per-sequence loop. forward_decode is only entered in
        # decode mode (see HybridLinearAttnBackend.forward dispatch), where each
        # request contributes exactly one token, so #tokens == #requests. Multi-
        # token-per-seq speculative paths (target_verify / draft_extend) go
        # through forward_extend instead. Assert the invariant so a future
        # routing change fails loudly rather than silently corrupting state.
        if self.kernel_dispatcher.supports_packed_decode:
            assert qkv.shape[0] == cache_indices.shape[0], (
                "KDA packed decode requires one token per sequence (T=1): "
                f"got {qkv.shape[0]} tokens for {cache_indices.shape[0]} requests."
            )
            return self.kernel_dispatcher.packed_decode(
                mixed_qkv=qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
            )

        q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d

        lower_bound = kwargs.get("lower_bound", getattr(layer, "lower_bound", None))

        result = self.kernel_dispatcher.decode(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            lower_bound=lower_bound,
        )

        # Track SSM/conv states for prefix caching during decode.
        # The kernel updates ssm_states in-place; copy the updated states to
        # persistent track slots so the radix cache can store them.
        if not forward_batch.forward_mode.is_target_verify():
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices
            )

        return result

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0].transpose(-1, -2)

        ssm_states = mamba_cache_params.temporal

        # Check if we are in TARGET_VERIFY mode
        is_target_verify = forward_batch.forward_mode.is_target_verify()

        # Handle TARGET_VERIFY mode where extend_prefix_lens might not be set
        if forward_batch.extend_prefix_lens is not None:
            has_initial_state = forward_batch.extend_prefix_lens > 0
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        else:
            # TARGET_VERIFY mode: infer from spec_info
            if forward_batch.spec_info is not None and hasattr(
                forward_batch.spec_info, "draft_token_num"
            ):
                bs = forward_batch.batch_size
                draft_token_num = forward_batch.spec_info.draft_token_num
                has_initial_state = torch.ones(
                    bs, dtype=torch.bool, device=mixed_qkv.device
                )
                extend_seq_lens_cpu = [draft_token_num] * bs
            else:
                raise RuntimeError(
                    "extend_prefix_lens is None but cannot infer from spec_info. "
                    "This should not happen in TARGET_VERIFY mode."
                )

        splits = [layer.q_dim, layer.k_dim, layer.v_dim]

        if is_target_verify:
            # In TARGET_VERIFY mode, use causal_conv1d_update similar to decode path
            # to properly save intermediate conv states for rollback after verification.
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            intermediate_state_cache = mamba_cache_params.intermediate_ssm

            seq_len = mixed_qkv.shape[0]
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num

            conv_state_indices = cache_indices[:batch_size]
            intermediate_state_indices = torch.arange(
                cache_indices.shape[0], dtype=torch.int32, device=cache_indices.device
            )

            retrieve_next_token = self.forward_metadata.retrieve_next_token
            retrieve_next_sibling = self.forward_metadata.retrieve_next_sibling
            retrieve_parent_token = self.forward_metadata.retrieve_parent_token

            # Reshape mixed_qkv: (seq_len, dim) -> (batch_size, dim, draft_token_num)
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)

            # Transpose intermediate_conv_window from (..., K-1, dim) to (..., dim, K-1)
            intermediate_conv_window_transposed = (
                intermediate_conv_window_cache.transpose(-1, -2)
            )

            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                layer.conv_weights,
                layer.bias,
                activation="silu",
                conv_state_indices=conv_state_indices,
                intermediate_conv_window=intermediate_conv_window_transposed,
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)

            q, k, v = mixed_qkv.split(splits, dim=-1)
        else:
            mixed_qkv_t = mixed_qkv.transpose(0, 1)
            if self.forward_metadata.has_mamba_track_mask:
                mixed_qkv_to_track = mixed_qkv_t[
                    :, self.forward_metadata.track_conv_indices
                ].transpose(0, 1)
                conv_states[self.forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track
                )

            q, k, v = mixed_qkv_t.split(splits, dim=0)
            q_conv_weight, k_conv_weight, v_conv_weight = layer.conv_weights.split(
                splits, dim=0
            )
            q_conv_state, k_conv_state, v_conv_state = conv_states.split(splits, dim=-2)
            if layer.bias is not None:
                q_bias, k_bias, v_bias = layer.bias.split(splits, dim=0)
            else:
                q_bias, k_bias, v_bias = None, None, None

            q = causal_conv1d_fn(
                q,
                q_conv_weight,
                q_bias,
                activation="silu",
                conv_states=q_conv_state,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=extend_seq_lens_cpu,
            ).transpose(0, 1)
            k = causal_conv1d_fn(
                k,
                k_conv_weight,
                k_bias,
                activation="silu",
                conv_states=k_conv_state,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=extend_seq_lens_cpu,
            ).transpose(0, 1)
            v = causal_conv1d_fn(
                v,
                v_conv_weight,
                v_bias,
                activation="silu",
                conv_states=v_conv_state,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=extend_seq_lens_cpu,
            ).transpose(0, 1)

        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d

        h = None
        if is_target_verify:
            core_attn_out = self.kernel_dispatcher.target_verify(
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices[:batch_size],
                query_start_loc=query_start_loc,
                intermediate_states_buffer=intermediate_state_cache,
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                cache_steps=draft_token_num,
                retrieve_parent_token=retrieve_parent_token,
                lower_bound=getattr(layer, "lower_bound", None),
            )
        else:
            core_attn_out, h = self.kernel_dispatcher.extend(
                q=q,
                k=k,
                v=v,
                g=a,
                beta=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                lower_bound=getattr(layer, "lower_bound", None),
            )

        # Track SSM states at chunk boundaries for prefix caching.
        # The h tensor from the FLA kernel contains per-chunk boundary states.
        if h is not None and not forward_batch.forward_mode.is_target_verify():
            self._track_mamba_state_extend(
                forward_batch, h, ssm_states, self.forward_metadata
            )

        return core_attn_out