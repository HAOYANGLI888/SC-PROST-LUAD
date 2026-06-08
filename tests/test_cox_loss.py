import torch

from models.survival_losses import cox_ph_loss


def test_cox_loss_is_finite_and_differentiable():
    risk = torch.tensor([1.2, 0.3, -0.5], requires_grad=True)
    durations = torch.tensor([2.0, 4.0, 6.0])
    events = torch.tensor([1.0, 1.0, 0.0])
    loss = cox_ph_loss(risk, durations, events)
    loss.backward()
    assert torch.isfinite(loss)
    assert risk.grad is not None
    assert torch.all(torch.isfinite(risk.grad))


def test_cox_loss_uses_shared_breslow_denominator_for_tied_events():
    scores = torch.tensor([0.0, 0.0, 0.0])
    durations = torch.tensor([5.0, 5.0, 2.0])
    events = torch.tensor([1.0, 1.0, 0.0])
    expected = torch.log(torch.tensor(2.0))
    assert torch.isclose(cox_ph_loss(scores, durations, events), expected)
