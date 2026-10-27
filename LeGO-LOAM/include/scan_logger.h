// scan_logger.h
//
// Header-only per-scan CSV logger for LeGO-LOAM instrumentation.
//
// Usage (per node):
//   #include "scan_logger.h"
//   static scanlog::ScanLogger g_log("features");   // -> $LEGO_LOG_DIR/features.csv
//   ...
//   g_log.set("n_corner_sharp", (double)cornerPointsSharp->points.size());
//   g_log.setInts("bear_corner", scanlog::bearingHistogram(*cornerPointsLessSharp, 36));
//   g_log.flushRow(timeScanCur);

#ifndef SCAN_LOGGER_H
#define SCAN_LOGGER_H

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cerrno>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace scanlog {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
inline bool mkdirRecursive(const std::string& path) {
  if (path.empty()) return false;
  std::string acc;
  size_t i = 0;
  if (path[0] == '/') { acc = "/"; i = 1; }
  while (i <= path.size()) {
    size_t j = path.find('/', i);
    if (j == std::string::npos) j = path.size();
    std::string part = path.substr(i, j - i);
    if (!part.empty()) {
      if (!acc.empty() && acc[acc.size() - 1] != '/') acc += "/";
      acc += part;
      if (::mkdir(acc.c_str(), 0775) != 0 && errno != EEXIST) return false;
    }
    i = j + 1;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------

class ScanLogger {
 public:
  // tag becomes the filename stem, e.g. "features" -> $LEGO_LOG_DIR/features.csv
  explicit ScanLogger(const std::string& tag)
      : enabled_(false), header_written_(false), schema_error_(false), rows_(0) {
    const char* dir = std::getenv("LEGO_LOG_DIR");
    if (dir == NULL || std::string(dir).empty()) {
      return;  // logging disabled; all subsequent calls are no-ops
    }
    dir_ = std::string(dir);
    if (dir_[0] != '/') {
      // roslaunch runs nodes from $ROS_HOME, so a relative LEGO_LOG_DIR will
      // not point where the caller thinks it does. Refuse rather than write
      // an orphan file somewhere nobody will look for it.
      std::fprintf(stderr,
          "[scanlog] FATAL: LEGO_LOG_DIR is relative ('%s'). Nodes launched by "
          "roslaunch run from $ROS_HOME, so this path will not resolve where "
          "you expect. Export an ABSOLUTE path. Logging disabled.\n",
          dir_.c_str());
      return;
    }
    if (!mkdirRecursive(dir_)) {
      std::fprintf(stderr,
          "[scanlog] FAILED to create directory %s (errno %d) -- "
          "logging disabled\n", dir_.c_str(), errno);
      return;
    }
    path_ = dir_ + "/" + tag + ".csv";
    file_.open(path_.c_str(), std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
      std::fprintf(stderr,
          "[scanlog] FAILED to open %s (errno %d) -- logging disabled. "
          "Check the path exists and is writable by this user.\n",
          path_.c_str(), errno);
      return;
    }
    file_ << std::setprecision(12);
    enabled_ = true;
    std::fprintf(stderr, "[scanlog] logging to %s\n", path_.c_str());
  }

  ~ScanLogger() {
    if (file_.is_open()) {
      file_.flush();
      file_.close();
    }
  }

  bool enabled() const { return enabled_; }

  // --- SCHEMA DECLARATION -------------------------------------------------
  // Declare every column this logger will ever write, in the node constructor,
  // BEFORE any row is flushed.
  void declare(const std::string& key) {
    if (!enabled_) return;
    declared_.insert(key);
  }

  void declareMatrix(const std::string& prefix, int rows, int cols) {
    if (!enabled_) return;
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c) {
        std::ostringstream k; k << prefix << "_r" << r << "c" << c;
        declared_.insert(k.str());
      }
  }

  void declareSymMatrix(const std::string& prefix, int n) {
    if (!enabled_) return;
    for (int r = 0; r < n; ++r)
      for (int c = r; c < n; ++c) {
        std::ostringstream k; k << prefix << "_r" << r << "c" << c;
        declared_.insert(k.str());
      }
  }

  void declareVector(const std::string& prefix, int n) {
    if (!enabled_) return;
    for (int i = 0; i < n; ++i) {
      std::ostringstream k;
      k << prefix << "_" << std::setfill('0') << std::setw(2) << i;
      declared_.insert(k.str());
    }
  }

  // True if any undeclared key has been seen. run_batch.sh should reject a run
  // whose log directory contains a SCHEMA_ERROR file (written automatically).
  bool schemaError() const { return schema_error_; }

  // --- scalar -------------------------------------------------------------
  void set(const std::string& key, double value) {
    if (!enabled_) return;
    row_[key] = value;
  }

  // --- dense matrix, row-major -------------------------------------------
  // Emits columns <prefix>_r0c0, <prefix>_r0c1, ... Accepts float or double.
  template <typename T>
  void setMatrix(const std::string& prefix, const T* data, int rows, int cols) {
    if (!enabled_ || data == NULL) return;
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        std::ostringstream k;
        k << prefix << "_r" << r << "c" << c;
        row_[k.str()] = static_cast<double>(data[r * cols + c]);
      }
    }
  }

  // --- symmetric matrix, upper triangle only ------------------------------
  // For a 6x6 this writes 21 columns instead of 36. 
  template <typename T>
  void setSymMatrix(const std::string& prefix, const T* data, int n) {
    if (!enabled_ || data == NULL) return;
    for (int r = 0; r < n; ++r) {
      for (int c = r; c < n; ++c) {
        std::ostringstream k;
        k << prefix << "_r" << r << "c" << c;
        row_[k.str()] = static_cast<double>(data[r * n + c]);
      }
    }
  }

  // --- integer vector (histograms) ----------------------------------------
  // Emits columns <prefix>_00, <prefix>_01, ...
  void setInts(const std::string& prefix, const std::vector<int>& v) {
    if (!enabled_) return;
    for (size_t i = 0; i < v.size(); ++i) {
      std::ostringstream k;
      k << prefix << "_" << std::setfill('0') << std::setw(2) << i;
      row_[k.str()] = static_cast<double>(v[i]);
    }
  }

  void setDoubles(const std::string& prefix, const std::vector<double>& v) {
    if (!enabled_) return;
    for (size_t i = 0; i < v.size(); ++i) {
      std::ostringstream k;
      k << prefix << "_" << std::setfill('0') << std::setw(2) << i;
      row_[k.str()] = v[i];
    }
  }

  // --- commit one scan ----------------------------------------------------
  // timestamp must be the ROS time of the scan being processed, in seconds,
  // so rows can be joined against the TUM trajectory and across nodes.
  void flushRow(double timestamp) {
    if (!enabled_) return;
    row_["t"] = timestamp;

    if (!header_written_) {
      columns_.clear();
      columns_.push_back("t");
      if (!declared_.empty()) {
        // Schema comes from the declarations. Any key present in this first
        // row but never declared is a bug and is reported below.
        for (std::set<std::string>::const_iterator it = declared_.begin();
             it != declared_.end(); ++it) {
          if (*it != "t") columns_.push_back(*it);
        }
      } else {
        std::fprintf(stderr,
            "[scanlog] WARNING: no columns were declared for %s. Falling back "
            "to freezing the schema on this first row. Any column produced by "
            "a code path that did not execute on this scan WILL BE LOST. "
            "Call declare*() in the node constructor.\n", path_.c_str());
        for (std::map<std::string, double>::const_iterator it = row_.begin();
             it != row_.end(); ++it) {
          if (it->first != "t") columns_.push_back(it->first);
        }
      }
      column_set_.clear();
      for (size_t i = 0; i < columns_.size(); ++i) column_set_.insert(columns_[i]);
      for (size_t i = 0; i < columns_.size(); ++i) {
        if (i) file_ << ",";
        file_ << columns_[i];
      }
      file_ << "\n";
      header_written_ = true;
    }

    // Strict check: an undeclared key means the schema is wrong, and the value
    // is about to be thrown away
    for (std::map<std::string, double>::const_iterator it = row_.begin();
         it != row_.end(); ++it) {
      if (column_set_.count(it->first) == 0) {
        std::fprintf(stderr,
            "[scanlog] SCHEMA ERROR in %s: key '%s' was never declared and its "
            "value is being discarded.\n", path_.c_str(), it->first.c_str());
        if (!schema_error_) {
          schema_error_ = true;
          std::string marker = dir_ + "/SCHEMA_ERROR";
          std::ofstream m(marker.c_str(), std::ios::out | std::ios::app);
          if (m.is_open()) {
            m << path_ << ": undeclared key '" << it->first << "'\n";
            m.close();
          }
        }
      }
    }

    const double nan_v = std::numeric_limits<double>::quiet_NaN();
    for (size_t i = 0; i < columns_.size(); ++i) {
      if (i) file_ << ",";
      std::map<std::string, double>::const_iterator it = row_.find(columns_[i]);
      double v = (it == row_.end()) ? nan_v : it->second;
      if (v != v) {
        file_ << "nan";
      } else {
        file_ << v;
      }
    }
    file_ << "\n";

    // Flush periodically so a crashed or killed run still leaves usable data.
    if ((++rows_ % 20) == 0) file_.flush();

    row_.clear();
  }

 private:
  bool enabled_;
  bool header_written_;
  bool schema_error_;
  unsigned long rows_;
  std::string path_;
  std::string dir_;
  std::ofstream file_;
  std::map<std::string, double> row_;      // sorted -> deterministic column order
  std::set<std::string> declared_;         // sorted -> deterministic column order
  std::set<std::string> column_set_;       // fast membership test
  std::vector<std::string> columns_;
};

// ---------------------------------------------------------------------------
// Cloud helpers (templated: work with any PCL-like cloud of points with .x/.y/.z)
// ---------------------------------------------------------------------------

// Which two axes span the horizontal plane, i.e. the plane the rover drives in.
//
// LeGO-LOAM's featureAssociation and mapOptmization operate in the LOAM
// "camera" convention, in which the sensor frame has been permuted so that
// z is forward, x is left and y is up. imageProjection publishes in the raw
// sensor frame where x is forward, y is left and z is up.
//
// VERIFY THIS EMPIRICALLY
enum HorizontalFrame {
  FRAME_LOAM_CAMERA = 0,  // horizontal plane = (z forward, x left), up = y
  FRAME_SENSOR = 1        // horizontal plane = (x forward, y left), up = z
};

inline void pickAxes(const double x, const double y, const double z,
                     int frame, double* fwd, double* lat, double* up) {
  if (frame == FRAME_SENSOR) {
    *fwd = x; *lat = y; *up = z;
  } else {
    *fwd = z; *lat = x; *up = y;
  }
}

// Histogram of feature bearings in the horizontal plane, over [-pi, pi).
template <typename CloudT>
std::vector<int> bearingHistogram(const CloudT& cloud, int nbins,
                                  int frame = FRAME_LOAM_CAMERA) {
  std::vector<int> h(nbins, 0);
  if (nbins <= 0) return h;
  const double two_pi = 6.283185307179586;
  for (size_t i = 0; i < cloud.points.size(); ++i) {
    double fwd, lat, up;
    pickAxes(cloud.points[i].x, cloud.points[i].y, cloud.points[i].z,
             frame, &fwd, &lat, &up);
    double b = std::atan2(lat, fwd);            // [-pi, pi]
    int bin = (int)std::floor((b + 3.14159265358979) / two_pi * nbins);
    if (bin < 0) bin = 0;
    if (bin >= nbins) bin = nbins - 1;
    h[bin]++;
  }
  return h;
}

// Histogram of feature elevation angles over [-pi/2, pi/2).
template <typename CloudT>
std::vector<int> elevationHistogram(const CloudT& cloud, int nbins,
                                    int frame = FRAME_LOAM_CAMERA) {
  std::vector<int> h(nbins, 0);
  if (nbins <= 0) return h;
  const double pi = 3.14159265358979;
  for (size_t i = 0; i < cloud.points.size(); ++i) {
    double fwd, lat, up;
    pickAxes(cloud.points[i].x, cloud.points[i].y, cloud.points[i].z,
             frame, &fwd, &lat, &up);
    double horiz = std::sqrt(fwd * fwd + lat * lat);
    double e = std::atan2(up, horiz);           // [-pi/2, pi/2]
    int bin = (int)std::floor((e + pi / 2.0) / pi * nbins);
    if (bin < 0) bin = 0;
    if (bin >= nbins) bin = nbins - 1;
    h[bin]++;
  }
  return h;
}

// Histogram of point ranges in metres, uniform bins over [0, max_range].
template <typename CloudT>
std::vector<int> rangeHistogram(const CloudT& cloud, int nbins,
                                double max_range = 50.0) {
  std::vector<int> h(nbins, 0);
  if (nbins <= 0 || max_range <= 0.0) return h;
  for (size_t i = 0; i < cloud.points.size(); ++i) {
    const double x = cloud.points[i].x;
    const double y = cloud.points[i].y;
    const double z = cloud.points[i].z;
    double r = std::sqrt(x * x + y * y + z * z);
    int bin = (int)std::floor(r / max_range * nbins);
    if (bin < 0) bin = 0;
    if (bin >= nbins) bin = nbins - 1;
    h[bin]++;
  }
  return h;
}

// Second moments of the feature point positions about their centroid.
// Returns 9 values: mean_x, mean_y, mean_z, cxx, cxy, cxz, cyy, cyz, czz.
template <typename CloudT>
std::vector<double> scatterMoments(const CloudT& cloud) {
  std::vector<double> out(9, std::numeric_limits<double>::quiet_NaN());
  const size_t n = cloud.points.size();
  if (n < 3) return out;
  double sx = 0, sy = 0, sz = 0;
  for (size_t i = 0; i < n; ++i) {
    sx += cloud.points[i].x;
    sy += cloud.points[i].y;
    sz += cloud.points[i].z;
  }
  const double mx = sx / n, my = sy / n, mz = sz / n;
  double cxx = 0, cxy = 0, cxz = 0, cyy = 0, cyz = 0, czz = 0;
  for (size_t i = 0; i < n; ++i) {
    const double dx = cloud.points[i].x - mx;
    const double dy = cloud.points[i].y - my;
    const double dz = cloud.points[i].z - mz;
    cxx += dx * dx; cxy += dx * dy; cxz += dx * dz;
    cyy += dy * dy; cyz += dy * dz; czz += dz * dz;
  }
  const double d = (double)n;
  out[0] = mx; out[1] = my; out[2] = mz;
  out[3] = cxx / d; out[4] = cxy / d; out[5] = cxz / d;
  out[6] = cyy / d; out[7] = cyz / d; out[8] = czz / d;
  return out;
}

}  // namespace scanlog

#endif  // SCAN_LOGGER_H
