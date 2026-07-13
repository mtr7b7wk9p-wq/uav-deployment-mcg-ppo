# Physical Service Capacity Design

## Goal

Replace the normalized distance-based service capacity in the resource-cognition environment with a physically interpretable air-to-ground link model. A scheduled task shall be served according to path loss, received power, same-band interference, noise, SINR, and Shannon rate.

## Scope Boundary

This stage keeps the existing task-level aggregate queue and discrete task-selection actions. It does not add user association, continuous power control, bandwidth allocation actions, or a new energy state model. Bandwidth, transmit power, and service duration are fixed scenario parameters.

The existing `mcg_ppo` coverage environment is unchanged. Only `ResourceCognitionEnv` uses the new physical service model.

## Units

- Queue length, arrival amount, and served data: Mbit;
- Bandwidth: MHz in configuration, converted to Hz in the rate calculation;
- Transmit power and noise power: W;
- Service duration: seconds;
- Path loss: dB;
- SINR: linear ratio;
- Rate: Mbit/s;
- Service capacity: Mbit per environment step.

## Channel Model

Reuse `AirToGroundChannelConfig` and `average_atg_path_loss_db` from `envs/channel.py`. For UAV `i` and task `q`:

```text
path_loss_db(i,q) = average air-to-ground path loss
channel_gain(i,q) = 10^(-path_loss_db(i,q) / 10)
received_power(i,q) = tx_power_w * channel_gain(i,q)
```

For all assigned tasks using the same band, the received power from other scheduled UAVs at task `q` is added as interference. The desired signal is the scheduled UAV's received power at its own task location.

```text
sinr(i,q) = received_power(i,q)
            / (noise_power_w + same_band_interference_w(i,q))
```

Shannon service rate and per-step capacity are:

```text
rate_mbps(i,q) = bandwidth_mhz * log2(1 + sinr(i,q))
capacity_mbit(i,q) = rate_mbps(i,q) * service_duration_s
```

Capacity is clipped by `cognition_max_service_per_step` and then by the task queue. Existing conflict penalties remain as a learning penalty, but same-band interference also reduces the actual service capacity.

## Energy Compatibility

Keep `remaining_time` as the current UAV resource proxy. A successful service still deducts the configured service duration from the remaining time. The existing `cognition_service_energy_cost` remains a reward/energy penalty proxy in this stage. Physical propulsion and radio power accounting are deferred until the service-capacity model is validated.

## Configuration

Add validated fields:

```python
cognition_bandwidth_mhz: float = 1.0
cognition_tx_power_w: float = 1.0
cognition_noise_power_w: float = 1e-13
cognition_service_duration_s: float = 1.0
cognition_channel_carrier_freq_ghz: float = 2.0
cognition_channel_los_a: float = 9.61
cognition_channel_los_b: float = 0.16
cognition_channel_eta_los_db: float = 1.0
cognition_channel_eta_nlos_db: float = 20.0
cognition_outage_sinr_threshold: float = 0.0
```

The outage threshold is a linear SINR threshold. If a scheduled link is below it, its service capacity is zero and it contributes to `service_outage_count`.

## Environment Interfaces

Add a focused helper in `ResourceCognitionEnv`:

```python
_physical_service_capacity(
    assignments: np.ndarray,
    conflict_counts: np.ndarray,
    use_truth: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

It returns per-agent capacity, SINR, path loss, and outage mask. `_evaluate_schedule` uses it for both hidden-truth evaluation and local-belief estimated utility. `_execute_scheduling` applies the returned capacity to queues and reports actual service.

## Metrics and Observation

Add info and episode metrics:

- `mean_path_loss_db`;
- `mean_sinr`;
- `mean_service_capacity`;
- `service_outage_count`;
- `service_outage_rate`;
- `total_interference_power_w`.

Do not expose hidden path loss, SINR, or interference directly in the local observation during this stage. The policy only sees the existing local link-quality proxy and learned queue/spectrum beliefs.

## Validation

Add failing contracts for:

1. A closer UAV has no lower path-loss capacity than a farther UAV under equal conditions.
2. Increasing noise decreases SINR and capacity.
3. A same-band interferer decreases the target link's SINR and service capacity.
4. A link below the outage threshold serves zero data and increments the outage count.
5. Actual served data remains bounded by queue, physical capacity, and the configured per-step limit.
6. The resource PPO one-update smoke test and legacy `mcg_ppo` smoke test still run.
