# Trajectory Data

The two CSV files contain 120-second excerpts of chronological position samples from two flight recordings.

```text
timestamp_s,x_m,y_m,z_m
```

- `timestamp_s`: seconds from the first row, beginning at `0.0`.
- `x_m`: local north position in metres.
- `y_m`: local east position in metres.
- `z_m`: local down position in metres.

The supplied excerpts are relatively clean and sampled at approximately 5 Hz. They do not contain the non-constant update spacing or spatial errors expected from real inputs. The supplied evaluator replays one selected excerpt exactly and linearly interpolates it internally to form the 50 Hz reference; it does not synthesize input errors.
