|            | explaination     | Value                  | possible                  |
| ------------------- | ---------------- | ---------------------- | ------------------------- |
| quantile            | 百分之多少的数据 | [0.95, 0.99]           | ok                        |
| rpc_0_client        | 1-8              | s[217948.1, 226872.76] | ok                        |
| rpc_0_server        | 4-5              | [217538.4, 225931.82]  | ok                        |
|                     |                  |                        |                           |
| rpc_1_net_req       | 1-4              | [272, 300]             | partly                    |
| rpc_1_net_resp      | 5-8              | [206, 264]             | partly                    |
|                     |                  |                        |                           |
| service_0_interfere |                  | 0.0                    | unknown                   |
| service_0_num_cores | 核的数量         | 1.0                    | ok                        |
| service_0_cpu_util  | cpu使用率        | 58.00                  | process_cpu_seconds_total |
| service_0_core_util | core使用率       | 53.21                  |                           |
| service_0_mem_util  | memory使用率     | 0.016                  | ok                        |
| service_0_netio_rd  | 网络读取         | 315.07                 |                           |
| service_0_netio_wr  | 网络写入         | 393.84                 |                           |
| service_0_blkio_rd  | disk读取         | 0.0                    |                           |
| service_0_blkio_wr  | disk写入         | 0.0                    | ok                        |
| service_0_ping      | ping相应时间？   | 0.044                  |                           |
| service_0_pkt_loss  | packet loss      | 0.0                    | ok                        |
|                     |                  |                        |                           |
| qps                 |                  | []                     |                           |

