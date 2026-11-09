# Enterprise Distributed File System Policy & Runbook

## Document Overview
This document specifies replication parameters, rack awareness rules, and incident remediation policies for the distributed storage cluster.

## Cluster Replication Rules
1. The default replication factor across all production data nodes is set to 3.
2. For mission-critical transactional logs, the replication factor must be strictly maintained at 5.
3. Under-replicated blocks are flagged by the HealthDaemon if replicas fall below the configured policy for more than 180 seconds.

## Rack Awareness Strategy
- The primary replica is always placed on a local rack node selected by proximity metric.
- The second replica is placed on a different node within the same local rack to minimize inter-switch bandwidth.
- The third replica is placed on a node located in an entirely distinct rack (Rack-B) to ensure fault tolerance against full-rack power failures.

## Incident Codes & Remediation
- **Error Code ERR_BLK_403**: Indicates a block permission mismatch during lease renewal. To remediate, flush node leases via `/opt/cluster/bin/leasectl --flush-all` and trigger secondary sync.
- **Error Code ERR_CORRUPT_701**: Checksum validation failed on block boundary. The Namenode isolates the affected chunk and requests reconstruction from healthy parity blocks.