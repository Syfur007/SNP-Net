"""XPU compatibility helpers for Lightning."""

from __future__ import annotations


def patch_lightning_xpu_parse_devices() -> None:
	"""Patch Lightning's XPU device parsing to use torch.xpu instead of CUDA.

	Lightning's built-in XPU accelerator uses CUDA-based parsing in some versions.
	This workaround makes `devices` parsing rely on torch.xpu so XPU can be used
	even when no CUDA devices are present.
	"""
	try:
		import torch
		from lightning.pytorch.accelerators.xpu import XPUAccelerator as LightningXPUAccelerator

		if not hasattr(torch, "xpu"):
			return

		def _parse_devices(devices):
			if devices in (None, "auto"):
				return list(range(torch.xpu.device_count()))
			if isinstance(devices, int):
				return list(range(devices))
			if isinstance(devices, str) and devices.isdigit():
				return list(range(int(devices)))
			if isinstance(devices, (list, tuple)):
				return list(devices)
			raise ValueError(f"Unsupported devices specification: {devices}")

		LightningXPUAccelerator.parse_devices = staticmethod(_parse_devices)
	except Exception:
		# If patching fails, fall back to Lightning's default behavior
		return

