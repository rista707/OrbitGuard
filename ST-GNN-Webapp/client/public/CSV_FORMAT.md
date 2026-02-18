# CSV Upload Format for Orbit-Guard

## File Requirements

- **Format**: CSV (Comma-Separated Values)
- **Rows**: Minimum 10 rows (representing timesteps T0-T9)
- **Columns**: 24 features per row
- **Header**: Optional (will be automatically detected and skipped)

## Column Order (24 Features)

The CSV must contain exactly 24 columns in the following order:

### Flow-Level Features (3)
1. `loss_ratio` - Packet loss ratio (0-1)
2. `throughput_Mbps` - Throughput in Megabits per second
3. `window_s` - Time window in seconds

### Routing Features (3)
4. `path_len` - Number of hops in the path
5. `includes_s_bad` - Binary indicator for bad satellite in path (0 or 1)
6. `path_changed` - Binary indicator for route change (0 or 1)

### ISL (Inter-Satellite Link) Features (5)
7. `isl_util_mean_path` - Mean ISL utilization on path (0-1)
8. `isl_util_max_path` - Maximum ISL utilization on path (0-1)
9. `isl_util_std_path` - Standard deviation of ISL utilization
10. `isl_delta_mean_path` - Delta in mean ISL utilization
11. `path_congestion_score` - Overall path congestion metric

### Engineered Features (13)
12. `throughput_anomaly` - Z-score normalized throughput deviation
13. `loss_spike` - Binary indicator for high loss >90% (0 or 1)
14. `zero_throughput` - Binary indicator for zero throughput (0 or 1)
15. `path_efficiency` - Inverse of path length: 1/(path_len+1)
16. `path_anomaly_score` - Combined metric: path_len × loss_ratio
17. `routing_instability` - Path change indicator (0 or 1)
18. `isl_congestion_level` - Categorical ISL congestion (0-3)
19. `isl_variance_ratio` - ISL utilization variance ratio
20. `bad_satellite_indicator` - Compromised satellite presence (0 or 1)
21. `high_path_len_indicator` - Abnormally long path indicator (0 or 1)
22. `low_throughput_indicator` - Abnormally low throughput indicator (0 or 1)
23. `congestion_loss_interaction` - Cross-layer: path_congestion_score × loss_ratio
24. `path_util_interaction` - Cross-layer: path_len × isl_util_mean_path

## Example CSV Format

### With Header (Recommended)
```csv
loss_ratio,throughput_Mbps,window_s,path_len,includes_s_bad,path_changed,isl_util_mean_path,isl_util_max_path,isl_util_std_path,isl_delta_mean_path,path_congestion_score,throughput_anomaly,loss_spike,zero_throughput,path_efficiency,path_anomaly_score,routing_instability,isl_congestion_level,isl_variance_ratio,bad_satellite_indicator,high_path_len_indicator,low_throughput_indicator,congestion_loss_interaction,path_util_interaction
0.95,1.2,5.0,4,0,1,0.35,0.80,0.25,0.01,2.5,0.15,1,0,0.20,3.8,1,2,1.14,0,0,1,2.375,1.4
0.96,1.3,5.0,4,0,0,0.36,0.81,0.26,0.02,2.6,0.18,1,0,0.20,3.84,0,2,1.12,0,0,1,2.496,1.44
... (8 more rows for T2-T9)
```

### Without Header
```csv
0.95,1.2,5.0,4,0,1,0.35,0.80,0.25,0.01,2.5,0.15,1,0,0.20,3.8,1,2,1.14,0,0,1,2.375,1.4
0.96,1.3,5.0,4,0,0,0.36,0.81,0.26,0.02,2.6,0.18,1,0,0.20,3.84,0,2,1.12,0,0,1,2.496,1.44
... (8 more rows for T2-T9)
```

## Sample File

A sample CSV file is available for download: [sample_sequence.csv](/sample_sequence.csv)

This file contains a sinkhole attack signature with 10 timesteps.

## Upload Process

1. Click the **"Upload CSV"** button in the Temporal Sequence Input section
2. Select your CSV file (must have `.csv` extension)
3. The system will:
   - Automatically detect and skip header row if present
   - Parse the first 10 rows as timesteps T0-T9
   - Load exactly 24 features per timestep
   - Pad with zeros if fewer than 24 columns
   - Truncate if more than 24 columns
4. All input fields will be automatically populated
5. You can then click **"Detect Attack"** to analyze the sequence

## Error Handling

- **"CSV must contain at least 10 rows"**: Your file has fewer than 10 data rows
- **"Failed to parse CSV file"**: The file format is incorrect or corrupted

## Tips

- Ensure numeric values are properly formatted (use `.` for decimals, not `,`)
- Binary features should be `0` or `1`
- Missing values will be treated as `0`
- You can upload a new file at any time to replace current data
- Use "Load Sample" button to see an example of properly formatted data

## Data Sources

This CSV format matches the output from the NS-3 LEO satellite network simulation pipeline:
1. Flow generation (UDP traffic metrics)
2. ISL state tracking (link utilization)
3. Routing path analysis (hop count, path changes)
4. Cross-layer feature engineering

For more details on how this data is generated, visit the **About** page.
