"""Inference-only model classes matching the architecture of
THE-TRAIN-LAB/OAI_Demo's exported VQ-VAE + criticality-aware-constellation bundles
(https://github.com/THE-TRAIN-LAB/OAI_Demo), independently reimplemented from its
public README/notebook description (no LICENSE file there, so this isn't a copy) -
same layer shapes are required for the published state dicts to load correctly, but
this file only includes what's needed to *load and decode* an existing bundle, not
train one (no classifier training, no VQ-VAE/constellation training, no RL agent
training loops - see the OAI_Demo notebook itself for those).

A "concept" is one of num_concepts=64 positional slots in the VQ-VAE's discrete
latent; each slot holds one index into a shared codebook of num_embeddings entries.
"Criticality"/importance here is a per-slot score (importance_net) - see
doc/SEMANTIC_QAM_DEMO.md for how this maps onto real PDSCH REs.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, inputs):
        flat = inputs.reshape(-1, self.embedding_dim)
        distances = (flat.square().sum(1, keepdim=True)
                     + self.embeddings.square().sum(1).unsqueeze(0)
                     - 2.0 * flat @ self.embeddings.T)
        indices = distances.argmin(dim=1)
        nearest = F.embedding(indices, self.embeddings).reshape_as(inputs)
        quantized = inputs + (nearest - inputs).detach()
        return quantized, indices


class ImportanceVQVAE(nn.Module):
    """784 (28x28 flattened) <-> num_concepts x latent_dim discrete latent."""

    def __init__(self, latent_dim=64, num_embeddings=256, num_concepts=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.num_concepts = num_concepts
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_concepts * latent_dim),
        )
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim)
        self.importance_net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_concepts), nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Flatten(), nn.Linear(num_concepts * latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 784), nn.Sigmoid(),
        )

    def encode(self, x):
        z = self.encoder(x).reshape(-1, self.num_concepts, self.latent_dim)
        zq, indices = self.vq_layer(z)
        return z, zq, indices.reshape(-1, self.num_concepts), self.importance_net(x)

    def decode(self, zq):
        return self.decoder(zq)


class TaskClassifier(nn.Module):
    """Downstream "semantic" task used to judge reconstruction quality by label
    preservation rather than pixel error - matches OAI_Demo's classifier_state.pt."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class LearnedConstellation(nn.Module):
    """A set of 2**num_bits_per_symbol complex points, zero-mean/unit-average-power
    normalized on every forward() call. For the "standard" scheme this stays at its
    Sionna-Gray-coded initialization; for "semantic" it's been moved (during OAI_Demo's
    own training) so higher-criticality bit groups land in less-confusable regions."""

    def __init__(self, num_bits_per_symbol):
        super().__init__()
        self.num_bits_per_symbol = int(num_bits_per_symbol)
        self.points_iq = nn.Parameter(torch.zeros(2 ** self.num_bits_per_symbol, 2))

    def forward(self):
        real = self.points_iq[:, 0] - self.points_iq[:, 0].mean()
        imag = self.points_iq[:, 1] - self.points_iq[:, 1].mean()
        power = torch.sqrt((real.square() + imag.square()).mean()).clamp_min(1e-12)
        return torch.complex(real / power, imag / power)


def integers_to_bits(values, width):
    powers = (1 << torch.arange(width, device=values.device, dtype=torch.long))
    return ((values.long().unsqueeze(-1) & powers) > 0).long()


def bits_to_integers(bits):
    powers = (1 << torch.arange(bits.shape[-1], device=bits.device, dtype=torch.long))
    return (bits.long() * powers).sum(dim=-1)


def squared_distance(rx, points):
    return (rx.reshape(-1, 1).real - points.reshape(1, -1).real).square() + \
           (rx.reshape(-1, 1).imag - points.reshape(1, -1).imag).square()


def reconstruct_from_received(vqvae, received_indices, selected_slots):
    """Scatter received codebook indices back into their original slots (unselected
    slots stay zero) and run the VQ-VAE decoder - the receiver-side half of the
    pipeline, independent of how the symbols got here (Sionna-simulated or real RF)."""
    batch, k = received_indices.shape
    embeddings = F.embedding(received_indices.long(), vqvae.vq_layer.embeddings)
    latent = torch.zeros(batch, vqvae.num_concepts, vqvae.latent_dim, device=embeddings.device)
    latent.scatter_(1, selected_slots.unsqueeze(-1).expand(-1, -1, vqvae.latent_dim), embeddings)
    return vqvae.decode(latent)


def encode_image_to_symbols(vqvae, constellation_points, image_784):
    """Encode a NEW image (not one of OAI_Demo's own 10 exported ones) into QAM symbols,
    using the same vqvae/constellation bundle already loaded for decode. Every bundle
    observed so far has num_embeddings == qam_order and k == num_concepts (all 64
    concepts always transmitted, nothing dropped by the importance-based top-k) - so
    each concept's codebook index directly IS its symbol index (see fetch_bundle.py's
    docstring), and there is nothing to actually *select*: this uses the concepts'
    natural order (selected_slots = arange(num_concepts)) rather than replicating
    OAI_Demo's own importance-based reordering, which the round trip is invariant to
    as long as encode and decode agree on the same order (they do, both here).

    image_784: array-like, shape (784,), float in [0,1] (flattened 28x28, matching
    OAI_Demo's own MNIST preprocessing).
    Returns (symbols_iq, symbol_indices, selected_slots) in the same shapes/dtypes as
    the bundle's own npz arrays, so this can substitute for the bundle's fixed 10
    images anywhere they're read from."""
    img = torch.as_tensor(image_784, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        _, _, indices, _ = vqvae.encode(img)
    symbol_indices = indices[0].numpy()
    symbols_iq = constellation_points[torch.as_tensor(symbol_indices, dtype=torch.long)].numpy()
    selected_slots = np.arange(vqvae.num_concepts)
    return symbols_iq, symbol_indices, selected_slots


def bit_error_rate(detected_symbols, true_symbol_indices, bits_per_symbol):
    """Fraction of mismatched bits between two arrays of symbol indices (same shape),
    each unpacked to bits_per_symbol bits - the BER counterpart to a raw symbol
    error rate, matching THE-TRAIN-LAB/Semantic-QAM's 'ber' metric."""
    detected = torch.as_tensor(detected_symbols, dtype=torch.long)
    true = torch.as_tensor(true_symbol_indices, dtype=torch.long)
    detected_bits = integers_to_bits(detected.reshape(-1), bits_per_symbol)
    true_bits = integers_to_bits(true.reshape(-1), bits_per_symbol)
    return (detected_bits != true_bits).float().mean().item()


def measure_semantic_quality(original, reconstructed, classifier, labels):
    """Port of THE-TRAIN-LAB/Semantic-QAM's measure_semantic_quality() (its notebook's
    TensorFlow version, ported to torch): task-based quality instead of pixel MSE -
    60% label preservation (classification accuracy) + 25% confidence preservation
    (how much the top-class probability shifts) + 15% distribution similarity
    (exp(-KL(original_probs || reconstructed_probs))). original/reconstructed are
    (batch, 784) in [0,1]; labels are (batch,) integer ground truth."""
    with torch.no_grad():
        pred_original = torch.softmax(classifier(original), dim=1)
        pred_reconstructed = torch.softmax(classifier(reconstructed), dim=1)

    label_preservation = (pred_reconstructed.argmax(1) == labels).float().mean().item()

    conf_original = pred_original.max(1).values
    conf_reconstructed = pred_reconstructed.max(1).values
    confidence_preservation = 1.0 - (conf_original - conf_reconstructed).abs().mean().item()

    eps = 1e-10
    kl_div = (pred_original * torch.log((pred_original + eps) / (pred_reconstructed + eps))).sum(1).mean().item()
    distribution_similarity = float(np.exp(-kl_div))

    semantic_quality = 0.6 * label_preservation + 0.25 * confidence_preservation + 0.15 * distribution_similarity
    return {
        "semantic_quality": semantic_quality,
        "label_preservation": label_preservation,
        "confidence_preservation": confidence_preservation,
        "distribution_similarity": distribution_similarity,
    }


def demap_and_reconstruct(rx_symbols_iq, constellation_points, selected_slots, k,
                           bits_per_concept, bits_per_symbol, payload_bit_length, vqvae):
    """Hard-decision nearest-point demodulation of rx_symbols_iq (whatever channel they
    actually went through - real RF here) against constellation_points, then decode
    back into codebook indices and an image. Mirrors OAI_Demo's semantic_qam_decode(),
    generalized to take the received symbols as a plain tensor rather than assuming
    they came from its own Sionna AWGN call."""
    rx = torch.as_tensor(rx_symbols_iq, dtype=torch.complex64)
    if rx.ndim == 1:
        rx = rx.unsqueeze(0)
    detected_symbols = squared_distance(rx, constellation_points).argmin(1).reshape(rx.shape)
    bits = integers_to_bits(detected_symbols, bits_per_symbol).reshape(rx.shape[0], -1)
    bits = bits[:, :payload_bit_length]
    recovered_indices = bits_to_integers(bits.reshape(rx.shape[0], k, bits_per_concept))
    slots = torch.as_tensor(selected_slots, dtype=torch.long)
    reconstruction = reconstruct_from_received(vqvae, recovered_indices, slots)
    return reconstruction.detach().numpy(), recovered_indices.numpy(), detected_symbols.numpy()
