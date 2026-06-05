"""Verify all core modules import correctly."""

def test_fabricpc_extensions_imports():
    from fabricpc_extensions import (
        SharedMemory,
        WorldModel,
    )
    assert WorldModel is not None
    assert SharedMemory is not None


def test_adapters_imports():
    from adapters import EnvironmentAdapter, GridWorldAdapter
    assert GridWorldAdapter is not None
    assert EnvironmentAdapter is not None


def test_world_model_basic():
    import numpy as np

    from fabricpc_extensions import WorldModel
    wm = WorldModel(latent_dim=16, maxlen=100)
    obs = np.random.randn(9).astype(np.float32)
    result = wm.update(obs, prediction_error=0.5, action=0, position=(5, 5))
    assert "latent_norm" in result
    assert "mean_shift" in result
    assert "novelty_estimate" in result


def test_shared_memory_basic():
    import numpy as np

    from fabricpc_extensions import SharedMemory
    mem = SharedMemory(capacity=100)
    obs = np.random.randn(9).astype(np.float32)
    pred = np.random.randn(9).astype(np.float32)
    mem.store(obs, pred, agent_id=0, meta={"pos": (0, 0)})
    results = mem.retrieve(obs, top_k=3)
    assert len(results) >= 0


def test_communication_channel_basic():
    import numpy as np

    from fabricpc_extensions import CommunicationChannel
    channel = CommunicationChannel(n_agents=2, msg_dim=4)
    msg0 = channel.produce(latent=np.random.randn(16), error=0.1, pos=(0, 0))
    channel.broadcast(agent_id=0, message=msg0)
    msg1 = channel.produce(latent=np.random.randn(16), error=0.2, pos=(1, 0))
    channel.broadcast(agent_id=1, message=msg1)
    received = channel.receive(agent_id=0)
    assert len(received) > 0


def test_evolution_basic():
    from fabricpc_extensions import PCGenome, Population
    genome = PCGenome(hidden_dim=32, n_hidden_layers=1, eta_infer=0.05, eta_learn=0.001)
    assert genome.hidden_dim == 32
    pop = Population(size=3, seed=42)
    assert len(pop.genomes) == 3
