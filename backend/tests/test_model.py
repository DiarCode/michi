"""Unit tests for DTS-GSSF model, data loader, and artifact store."""

import numpy as np
import torch

from backend.ml.model import DTSGSSF, GatedSSMBlock, GraphPropagation, LoRALinear, nb_nll, softplus


class TestLoRALinear:
    def test_forward_shape(self):
        layer = LoRALinear(32, 16, r=4)
        x = torch.randn(2, 32)
        y = layer(x)
        assert y.shape == (2, 16)

    def test_lora_parameters(self):
        layer = LoRALinear(32, 16, r=8)
        params = list(layer.lora_parameters())
        assert len(params) == 2
        assert params[0].shape == (8, 32)
        assert params[1].shape == (16, 8)

    def test_zero_r_disables_lora(self):
        layer = LoRALinear(32, 16, r=0)
        assert layer.A is None
        assert layer.B is None
        x = torch.randn(2, 32)
        y = layer(x)
        assert y.shape == (2, 16)


class TestGatedSSMBlock:
    def test_forward_shape(self):
        block = GatedSSMBlock(d_in=7, d_model=64, dropout=0.0, lora_r=4)
        x = torch.randn(1, 10, 5, 7)
        out = block(x)
        assert out.shape == (1, 5, 64)

    def test_output_is_normalized(self):
        block = GatedSSMBlock(d_in=7, d_model=32, dropout=0.0, lora_r=4)
        x = torch.randn(1, 5, 3, 7)
        out = block(x)
        assert out.mean().abs() < 2.0


class TestGraphPropagation:
    def test_forward_shape(self):
        N = 5
        A = np.eye(N, dtype=np.float32)
        A[0, 1] = A[1, 0] = 1.0
        gp = GraphPropagation(N=N, d=32, A_phys=A, K=2)
        h = torch.randn(1, N, 32)
        out = gp(h)
        assert out.shape == (1, N, 32)

    def test_adaptive_adj_shape(self):
        N = 5
        A = np.eye(N, dtype=np.float32)
        gp = GraphPropagation(N=N, d=32, A_phys=A, K=2)
        A_adp = gp.adaptive_adj()
        assert A_adp.shape == (N, N)
        row_sums = A_adp.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(N), atol=1e-4)


class TestDTSGSSF:
    def test_forward_shape(self):
        N = 5
        A = np.eye(N, dtype=np.float32)
        model = DTSGSSF(N=N, F_in=7, n_series=N, n_agg=3, A_phys=A, d_model=32, horizon=4, K=2, lora_r=4, dropout=0.0)
        x = torch.randn(1, 24, N, 7)
        mu, kappa = model(x)
        assert mu.shape[0] == 1
        assert mu.shape[2] == N + 3
        assert kappa.shape == ()

    def test_output_positive(self):
        N = 3
        A = np.eye(N, dtype=np.float32)
        model = DTSGSSF(N=N, F_in=7, n_series=N, n_agg=2, A_phys=A, d_model=16, horizon=4, dropout=0.0)
        x = torch.randn(1, 12, N, 7)
        mu, kappa = model(x)
        assert (mu > 0).all()
        assert kappa > 0

    def test_freeze_unfreeze(self):
        N = 3
        A = np.eye(N, dtype=np.float32)
        model = DTSGSSF(N=N, F_in=7, n_series=N, n_agg=2, A_phys=A, d_model=16, horizon=4, dropout=0.0)
        model.freeze_base_for_adaptation()
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        total = sum(1 for p in model.parameters())
        assert trainable > 0
        assert trainable < total

        model.unfreeze_all()
        trainable_after = sum(1 for p in model.parameters() if p.requires_grad)
        assert trainable_after == total


class TestNBNLL:
    def test_loss_positive(self):
        y = torch.tensor([10.0, 20.0, 5.0])
        mu = torch.tensor([12.0, 18.0, 6.0])
        kappa = torch.tensor(5.0)
        loss = nb_nll(y, mu, kappa)
        # nb_nll returns per-element losses; all should be positive (NLL)
        assert (loss > 0).all()

    def test_loss_shape(self):
        y = torch.randn(2, 4, 10).abs()
        mu = torch.randn(2, 4, 10).abs() + 0.1
        kappa = torch.tensor(3.0)
        loss = nb_nll(y, mu, kappa.expand_as(mu))
        # Returns per-element tensor same shape as inputs
        assert loss.shape == y.shape

    def test_zero_y(self):
        y = torch.zeros(3)
        mu = torch.ones(3)
        kappa = torch.tensor(5.0)
        loss = nb_nll(y, mu, kappa)
        assert torch.isfinite(loss).all()


class TestSoftplus:
    def test_positive_input(self):
        x = torch.tensor(5.0)
        result = softplus(x)
        assert result > 0

    def test_negative_input(self):
        x = torch.tensor(-5.0)
        result = softplus(x)
        assert result > 0
