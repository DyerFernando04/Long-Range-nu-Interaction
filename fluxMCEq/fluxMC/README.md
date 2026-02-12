# How to use

Use: python3 flux.py

## Description

This script generates numu, nue, antinumu, and antinue fluxes with MCEq with full sky coverage (0-180).
Internally it uses a 0-90 degree range to compute and fold. When theta=120 degree, it uses theta_ref=60 degree and
saves the result in theta=120 degree.
