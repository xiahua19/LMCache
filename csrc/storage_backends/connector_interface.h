// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "connector_types.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace lmcache {
namespace connector {

/*
instantiated in connector_base.h and further overridden by custom storage
connectors
*/
class IStorageConnector {
 public:
  virtual ~IStorageConnector() = default;

  virtual int event_fd() const = 0;

  /*
  submit a batch GET operation

  retrieves values for multiple keys in parallel. work is automatically divided
  among worker threads (tiling). returns a single future_id for the entire
  batch.

  args:
    keys: vector of key strings to retrieve
    bufs: vector of buffer pointers (must be writable, size ==
  batch_chunk_num_bytes for each buffer) lens: vector of buffer sizes (each must
  equal batch_chunk_num_bytes) batch_chunk_num_bytes: expected size of each
  value (optimization: avoids parsing)

  returns:
    uint64_t: future id for tracking this batch operation
  */
  virtual uint64_t submit_batch_get(const std::vector<std::string>& keys,
                                    const std::vector<void*>& bufs,
                                    const std::vector<size_t>& lens,
                                    size_t batch_chunk_num_bytes) = 0;

  /*
  submit a batch SET operation

  stores values for multiple keys in parallel. work is automatically divided
  among worker threads (tiling). returns a single future_id for the entire
  batch.

  args:
    keys: vector of key strings to store
      bufs: vector of buffer pointers (must be readable, size ==
  batch_chunk_num_bytes for each buffer) lens: vector of buffer sizes (each must
  equal batch_chunk_num_bytes) batch_chunk_num_bytes: size of each value
  (optimization: avoids parsing)

  returns:
    uint64_t: future id for tracking this batch operation
  */
  virtual uint64_t submit_batch_set(const std::vector<std::string>& keys,
                                    const std::vector<void*>& bufs,
                                    const std::vector<size_t>& lens,
                                    size_t batch_chunk_num_bytes) = 0;

  /*
  submit a batch EXISTS operation

  checks existence of multiple keys in parallel. work is automatically divided
  among worker threads (tiling). returns a single future_id for the entire
  batch.

  args:
    keys: vector of key strings to check

  returns:
    uint64_t: future id for tracking this batch operation
    completion will contain result_bytes vector with 0/1 for each key
  */
  virtual uint64_t submit_batch_exists(
      const std::vector<std::string>& keys) = 0;

  /*
  submit a batch DELETE operation

  deletes multiple keys in parallel. work is automatically divided
  among worker threads (tiling). returns a single future_id for the entire
  batch.

  args:
    keys: vector of key strings to delete

  returns:
    uint64_t: future id for tracking this batch operation
    completion will contain result_bytes vector with 0/1 for each key
    (1 = deleted, 0 = not found)
  */
  virtual uint64_t submit_batch_delete(
      const std::vector<std::string>& keys) = 0;

  /*
  drain all available completions

  this is the ONLY method that should be called when the eventfd becomes
  readable. it automatically drains the eventfd and returns all available
  completions.

  returns:
    std::vector<Completion>: all completions ready since last drain
  */
  virtual std::vector<Completion> drain_completions() = 0;

  /*
  shutdown the connector and cleanup resources

  this method:
  1. signals all worker threads to stop
  2. wakes up blocked workers
  3. shuts down network connections
  4. joins all worker threads
  5. closes eventfd
  6. clears pending requests and completions

  idempotent: safe to call multiple times (subsequent calls are no-ops)
  */
  virtual void close() = 0;

  /*
  Health interface (optional, mirrors L2AdapterInterface).

  The defaults say "not provided", so connectors written before these existed
  keep working unchanged.
  */

  // whether probe_health() is implemented
  virtual bool supports_health_probe() const { return false; }

  /*
  check that the backend answers and that this connector's connections are
  usable. blocks the calling thread for at most about timeout_ms; the caller
  must not hold any lock a worker needs. does not touch any data buffer.
  */
  virtual HealthProbe probe_health(int timeout_ms) {
    (void)timeout_ms;
    throw std::runtime_error("probe_health is not supported");
  }

  // whether reconnect() is implemented
  virtual bool supports_reconnect() const { return false; }

  /*
  re-establish every worker's connection without destroying the connector,
  waiting at most about timeout_ms for the new connections.

  on normal return every worker holds a new connection. every request
  submitted before the call either completed or is completed as a failure
  (store unsuccessful, empty per-key results) through the usual completion
  queue, and no worker touches the buffer of a request it has reported as
  finished. throws if the connections could not be restored; the caller may
  call again.
  */
  virtual void reconnect(int timeout_ms) {
    (void)timeout_ms;
    throw std::runtime_error("reconnect is not supported");
  }
};

}  // namespace connector
}  // namespace lmcache
