// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "connector_base.h"
#include "connector_interface.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace lmcache {
namespace connector {
namespace pybind_utils {

/*
these utilities:
1. convert between cpp and python types
2. release GIL immediately after extracting buffer pointers

example usage (see `redis/pybind.cpp`):
  py::class_<MyConnector>(m, "MyConnectorClient")
      .def(py::init<...>(), ...)
      LMCACHE_BIND_CONNECTOR_METHODS(MyConnector);
*/

#define LMCACHE_BIND_CONNECTOR_METHODS(ConnectorType)                      \
  .def("event_fd", &ConnectorType::event_fd)                               \
      .def("submit_batch_get",                                             \
           lmcache::connector::pybind_utils::bind_submit_batch_get<        \
               ConnectorType>(),                                           \
           py::arg("keys"), py::arg("memoryviews"))                        \
      .def("submit_batch_set",                                             \
           lmcache::connector::pybind_utils::bind_submit_batch_set<        \
               ConnectorType>(),                                           \
           py::arg("keys"), py::arg("memoryviews"))                        \
      .def("submit_batch_exists",                                          \
           lmcache::connector::pybind_utils::bind_submit_batch_exists<     \
               ConnectorType>(),                                           \
           py::arg("keys"))                                                \
      .def("submit_batch_delete",                                          \
           lmcache::connector::pybind_utils::bind_submit_batch_delete<     \
               ConnectorType>(),                                           \
           py::arg("keys"))                                                \
      .def("drain_completions",                                            \
           lmcache::connector::pybind_utils::bind_drain_completions<       \
               ConnectorType>())                                           \
      .def("supports_health_probe", &ConnectorType::supports_health_probe) \
      .def("probe_health",                                                 \
           lmcache::connector::pybind_utils::bind_probe_health<            \
               ConnectorType>(),                                           \
           py::arg("timeout_ms"))                                          \
      .def("supports_reconnect", &ConnectorType::supports_reconnect)       \
      .def("reconnect", &ConnectorType::reconnect, py::arg("timeout_ms"),  \
           py::call_guard<py::gil_scoped_release>())                       \
      .def("close", &ConnectorType::close,                                 \
           py::call_guard<py::gil_scoped_release>())

template <typename ConnectorType>
auto bind_submit_batch_get() {
  return [](ConnectorType& self, const std::vector<std::string>& keys,
            py::list memviews) {
    if (keys.size() != memviews.size()) {
      throw std::runtime_error("keys and memviews size mismatch");
    }
    if (keys.empty()) {
      throw std::runtime_error("keys list is empty");
    }

    // extract all buffer info under GIL
    std::vector<void*> bufs;
    std::vector<size_t> lens;
    bufs.reserve(keys.size());
    lens.reserve(keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
      py::memoryview mv = memviews[i].cast<py::memoryview>();
      py::buffer_info info = py::buffer(mv).request();
      bufs.push_back(info.ptr);
      lens.push_back(static_cast<size_t>(info.size));
    }

    // use the first buffer's size as batch_chunk_num_bytes (all must match)
    size_t batch_chunk_num_bytes = lens[0];

    py::gil_scoped_release release;
    return self.submit_batch_get(keys, bufs, lens, batch_chunk_num_bytes);
  };
}

template <typename ConnectorType>
auto bind_submit_batch_set() {
  return [](ConnectorType& self, const std::vector<std::string>& keys,
            py::list memviews) {
    if (keys.size() != memviews.size()) {
      throw std::runtime_error("keys and memviews size mismatch");
    }
    if (keys.empty()) {
      throw std::runtime_error("keys list is empty");
    }

    // extract all buffer info under GIL
    std::vector<void*> bufs;
    std::vector<size_t> lens;
    bufs.reserve(keys.size());
    lens.reserve(keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
      py::memoryview mv = memviews[i].cast<py::memoryview>();
      py::buffer_info info = py::buffer(mv).request();
      bufs.push_back(info.ptr);
      lens.push_back(static_cast<size_t>(info.size));
    }

    // use the first buffer's size as batch_chunk_num_bytes (all must match)
    size_t batch_chunk_num_bytes = lens[0];

    py::gil_scoped_release release;
    return self.submit_batch_set(keys, bufs, lens, batch_chunk_num_bytes);
  };
}

template <typename ConnectorType>
auto bind_submit_batch_exists() {
  return [](ConnectorType& self, const std::vector<std::string>& keys) {
    py::gil_scoped_release release;
    return self.submit_batch_exists(keys);
  };
}

template <typename ConnectorType>
auto bind_submit_batch_delete() {
  return [](ConnectorType& self, const std::vector<std::string>& keys) {
    py::gil_scoped_release release;
    return self.submit_batch_delete(keys);
  };
}

template <typename ConnectorType>
auto bind_drain_completions() {
  return [](ConnectorType& self) {
    // call cpp method without holding GIL
    std::vector<Completion> completions;
    {
      py::gil_scoped_release release;
      completions = self.drain_completions();
    }

    // convert results to python objects (requires GIL)
    py::list result;
    for (auto& c : completions) {
      // convert result_bytes to python list of bools if not empty
      py::object bools_obj = py::none();
      if (!c.result_bytes.empty()) {
        py::list bools_list;
        for (uint8_t b : c.result_bytes) {
          bools_list.append(bool(b));
        }
        bools_obj = bools_list;
      }

      // return tuple: (future_id, ok, error, result_bools)
      result.append(py::make_tuple(c.future_id, c.ok, c.error, bools_obj));
    }

    return result;
  };
}

// probe_health(timeout_ms) -> (status, reason), status one of "ok",
// "unreachable", "reconnect_needed". Waits for the backend, so the GIL is
// released for the whole call.
template <typename ConnectorType>
auto bind_probe_health() {
  return [](ConnectorType& self, int timeout_ms) {
    HealthProbe probe;
    {
      py::gil_scoped_release release;
      probe = self.probe_health(timeout_ms);
    }
    const char* status = "ok";
    if (probe.status == HealthStatus::UNREACHABLE) {
      status = "unreachable";
    } else if (probe.status == HealthStatus::RECONNECT_NEEDED) {
      status = "reconnect_needed";
    }
    return py::make_tuple(status, probe.reason);
  };
}

inline WorkerPoolConfig parse_per_op_workers(const py::object& obj) {
  if (obj.is_none()) {
    return {};
  }
  py::dict d = obj.cast<py::dict>();
  std::unordered_map<std::string, int> per_op;
  for (auto& [key, value] : d) {
    std::string key_str = key.cast<std::string>();
    int count = value.cast<int>();
    if (count <= 0) {
      throw std::runtime_error("per_op_workers['" + key_str +
                               "'] must be a positive integer");
    }
    per_op[key_str] = count;
  }
  return {std::move(per_op)};
}

}  // namespace pybind_utils
}  // namespace connector
}  // namespace lmcache
