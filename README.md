# UBItelematics

## Overview
This repository contains the source code, simulation data, and for the paper **PAPER TITLE**. It implements a dynamic pricing framework based on the logit-reduced mixture-of-experts model ([LRMoE.jl](https://actsci.utstat.utoronto.ca/LRMoE.jl/stable/)) designed for usage-based auto insurance pricing. The primary languages used in this repository are [Julia](https://julialang.org/) and [R](https://www.r-project.org/).

To enhance usability, simplify integration into other workflows, and ensure consistent execution, the code is packaged into a lightweight, easy-to-install module. *(building of this repository is still in progress, expected to be completed by the end of 2025)*

## Data
The dataset `SimUBIDataset` is a cleaned simulation driving behavior dataset containing daily-summarized statistics for **X** drivers, mirroring their long-term driving behavior in a 91 days time window.

| Variable             | Description                |
|----------------------|-------------------------------|
| `DayID`              | -                      |
| `DriverID`           | -                      |
| `total_triplength`   | Daily total driving duration in hours                    |
| `total_tripdistance` | Daily total driving distance in kilometers             |
| `hbrk`               | Daily sum of harsh braking events                    |
| `turn`               | Daily sum of sharp turning events                      |
| `numtrip`            | Daily sum of number of trips                      |
| `peakprop`           | Proportion of time travelled during rush hours                  |
| `avr_hbrk`           | `hbrk`/`total_tripdistance`                |
| `avr_turn`           | `turn`/`total_tripdistance`                  |
| `notripdays`         | accumulated sum of days without trips                    |
| `notripday_prop`     | Proportion of accumulated sum of days without trips up to date                |
| `accidents`          | Number of accidents occured in the latest 365 days                     |
| `age`                | Age of the driver                    |
## Implementation
To initialize, use [setup.jl](https://github.com/alanaguo/UBItelematics/edit/main/script/setup.jl) to set up the main module for implementation.

## Acknowledgements
