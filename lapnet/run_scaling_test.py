import jax
import numpy as np
from lapnet import (
    hamiltonian,
)
from lapnet import networks
from lapnet.train import (
    init_electrons,
)
from lapnet.utils.system import pyscf_mol_to_internal_representation
import pyscf.gto
import base_config
import jax.numpy as jnp
import timeit
import argparse
import jaxlib.xla_extension


def get_timing(expression, n_repeat=10):
    timer = timeit.Timer(expression, globals=globals())
    n_timeit_reps = timer.autorange()[0]
    timings = timer.repeat(repeat=n_repeat, number=n_timeit_reps)
    return min(timings) / n_timeit_reps


def get_cumulene(n_carbon: int, use_ecp):
    bond_length = 2.53222
    R = np.zeros([n_carbon + 4, 3])
    R[:-4, :] = np.arange(n_carbon)[:, None] * np.array([bond_length, 0, 0])[None, :]
    R_H = np.array([[-1.02612, 1.77728, 0.0], [-1.02612, -1.77728, 0.0]])
    R[-4:-2] = R_H
    R[-2:] = R[n_carbon - 1] - R_H
    if use_ecp:
        Z = np.array([4] * n_carbon + [1, 1, 1, 1])
    else:
        Z = np.array([6] * n_carbon + [1, 1, 1, 1])

    n_el = sum(Z)
    assert n_el % 2 == 0
    mol = pyscf.gto.M(atom=[(z, r) for z, r in zip(Z, R)], unit="bohr", basis="sto-6g")
    mol.build()
    return (
        mol,
        jnp.array(R, jnp.float32),
        jnp.array(Z, jnp.int32),
        (n_el // 2, n_el // 2),
    )


def get_config():
    return base_config.default()


def vmap_jit_await(f):
    f = jax.vmap(f, in_axes=(None, 0))
    f = jax.jit(f)
    return lambda p, r: jax.block_until_ready(f(p, r))


def loop_network(network, n_reps):
    def looped_network_fn(params, data):
        def body_fn(i, r):
            logpsi = network(params, r)
            return r + 1e-12 * logpsi

        return jax.lax.fori_loop(0, n_reps, body_fn, data)

    return looped_network_fn


def get_scaling(
    system,
    system_size,
    batch_size,
    use_ecp,
    model,
    n_determinants,
    n_iterations,
):
    """Runs training loop for QMC.

    Args:
      cfg: ConfigDict containing the system and training parameters to run on. See
        base_config.default for more details.
    Raises:
      ValueError: if an illegal or unsupported value in cfg is detected.
    """
    # Device logging
    assert jax.device_count() == 1, "Only single device is supported."

    cfg = get_config()
    if system == "cumulene":
        mol, atoms, charges, nspins = get_cumulene(system_size, use_ecp=use_ecp)
    else:
        raise ValueError(f"Unsupported system: {system}")
    cfg.update(pyscf_mol_to_internal_representation(mol))
    cfg.network.name = model
    cfg.network.detnet.determinants = n_determinants
    if model.lower() in ["psiformer", "lapnet"]:
        cfg.network.detnet.hidden_dims = ((256, 4),) * 4
    elif model.lower() == "ferminet":
        cfg.network.detnet.hidden_dims = ((256, 16),) * 4
    else:
        raise ValueError(f"Unsupported model: {model}")

    key = jax.random.PRNGKey(1234)
    rng_params, rng_data = jax.random.split(key)

    data = init_electrons(
        rng_data,
        cfg.system.molecule,
        cfg.system.electrons,
        batch_size,
        init_width=cfg.mcmc.init_width,
    )

    (network_init, signed_network, _, _) = networks.network_provider(cfg)(  # type: ignore
        atoms,
        nspins,
        charges,
    )

    network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]  # type: networks.LogWaveFuncLike

    local_energy = hamiltonian.local_energy(
        f=signed_network,
        atoms=atoms,
        charges=charges,
        nspins=nspins,
        use_scan=False,
        forward_laplacian=True,
        with_rng=False,
    )

    network = loop_network(network, n_iterations)
    batch_network = vmap_jit_await(network)
    batch_energy = vmap_jit_await(local_energy)
    params = network_init(rng_params)

    # compile
    batch_network(params, data)
    batch_energy(params, data)

    # Run for timings
    t_psi = get_timing(lambda: batch_network(params, data)) / n_iterations
    t_energy = get_timing(lambda: batch_energy(params, data))
    return dict(
        n_el_core=2 * system_size if use_ecp else 0,
        n_el=mol.nelectron,
        t_wf_full=t_psi,
        t_E_kin=t_energy,
    )


if __name__ == "__main__":
    default_system_sizes = np.unique(np.round(np.geomspace(16, 256, 25)).astype(int))

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use-ecp", action="store_true", default=False)
    parser.add_argument("--system", type=str, default="cumulene")
    parser.add_argument("--system_sizes", nargs="+", type=int, default=default_system_sizes)
    parser.add_argument("--n_iterations", type=int, default=250)
    parser.add_argument(
        "--model",
        type=str,
        default="ferminet",
        choices=["ferminet", "lapnet", "psiformer"],
    )
    parser.add_argument("--n_determinants", type=int, default=4)
    parser.add_argument("-o", "--output", type=str, default="timings.txt")

    args = parser.parse_args()

    batch_size = args.batch_size
    for system_size in sorted(args.system_sizes):
        while batch_size > 1:
            try:
                settings = dict(
                    system=args.system,
                    system_size=system_size,
                    batch_size=batch_size,
                    use_ecp=args.use_ecp,
                    model=args.model,
                    n_determinants=args.n_determinants,
                    n_iterations=args.n_iterations,
                )
                results = settings | get_scaling(**settings)
                with open(args.output, "a") as f:
                    f.write(str(results) + "\n")
                    print(results)
                break
            except jaxlib.xla_extension.XlaRuntimeError:
                batch_size //= 2
                print(f"OOM: reducing batch size. New batch-size: {batch_size}", flush=True)
        else:
            # This is only reached if the while loop completes without breaking
            print(f"Failed to run system size {system_size} with batch size 1. Exiting", flush=True)
            break
