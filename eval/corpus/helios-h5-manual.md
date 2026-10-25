# Helios H5 Hybrid Inverter - Technical Manual

## 1. Overview

The Helios H5 is a 5 kW single-phase hybrid solar inverter manufactured by Solaris Dynamics.
It combines a photovoltaic (PV) inverter, a battery charger and an energy management system in
one enclosure. The H5 was introduced in March 2024 and replaces the Helios H4.

## 2. Electrical specifications

| Parameter | Value |
|---|---|
| Rated AC output power | 5,000 W |
| Maximum PV input power | 7,500 W |
| Maximum DC input voltage | 600 V |
| MPPT voltage range | 120 V - 550 V |
| Number of MPPT trackers | 2 |
| Maximum input current per tracker | 16 A |
| Peak efficiency | 98.2 % |
| European weighted efficiency | 97.4 % |
| Battery voltage range | 40 V - 60 V (48 V nominal) |
| Maximum charge/discharge current | 100 A |

## 3. Installation

The inverter must be mounted vertically on a non-flammable wall. Keep at least 30 cm of clearance
above and below the unit and 20 cm on each side. The enclosure is rated IP65, so outdoor
installation is allowed, but direct sunlight reduces output because the H5 starts derating at
an ambient temperature of 45 degrees Celsius. The operating temperature range is -25 to 60
degrees Celsius.

Use copper cables with a minimum cross-section of 6 mm2 for the AC connection. An external
40 A circuit breaker is required between the inverter and the grid.

## 4. Operating modes

- **Self-consumption**: PV power supplies household loads first, surplus charges the battery,
  and remaining surplus is exported to the grid.
- **Time-of-use**: the battery charges from the grid during off-peak hours configured in the
  Helios app and discharges during peak hours.
- **Backup**: when the grid fails, the H5 switches to off-grid mode in under 10 milliseconds
  and powers the backup circuit up to 5,000 W.

## 5. Error codes

| Code | Meaning | Action |
|---|---|---|
| E01 | Grid over-voltage | Check grid voltage; contact the utility if it exceeds 253 V |
| E07 | Insulation resistance too low | Inspect PV strings for damaged insulation |
| E12 | Battery communication lost | Check the CAN cable between battery and inverter |
| E21 | Over-temperature | Improve ventilation; the unit restarts automatically |

## 6. Warranty

The standard warranty is 10 years from the date of installation. It can be extended to 15 years
by registering the product in the Helios app within 90 days of installation.
