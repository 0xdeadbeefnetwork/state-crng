# OpenCL Cryptographically Secure Random Number Generator (CRNG)

This project implements a cryptographically secure random number generator (CRNG) using OpenCL. The generator uses entropy from both user-supplied input and time-based seeds, ensuring high-quality, unpredictable randomness. The Python script tests and demonstrates the CRNG implementation.

## Features
- **Time-based seeding**: Incorporates system time (in nanoseconds) to add an extra layer of entropy.
- **SHA-256 hashing**: The internal state is updated and mixed using SHA-256, ensuring cryptographic security.
- **Feedback mechanism**: Random output is fed back into the state to further enhance unpredictability.
- **Portable**: Designed to work with any OpenCL-capable hardware.

## Requirements

- Python 3.x
- [PyOpenCL](https://pypi.org/project/pyopencl/) (`pip install pyopencl`)
- OpenCL-capable device (CPU/GPU)

## Setup

1. **Install dependencies**:
   Ensure that you have `pyopencl` installed:
   ```bash
   pip install pyopencl
