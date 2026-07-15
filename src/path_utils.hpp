#pragma once
#include <filesystem>
#include <string>

// ONNX Runtime's Ort::Session (and SetOptimizedModelFilePath) expect a
// path in the platform's "native" character type: `const wchar_t*` on
// Windows, `const char*` on Linux/macOS.
//
// The old code built this manually with:
//   std::wstring w(str.begin(), str.end());
// which widens each byte one-to-one. That's a Windows-only construct
// (it doesn't compile as the "native path" on POSIX at all, since there
// `ORTCHAR_T` is plain `char`), and even on Windows it silently mangles
// any non-ASCII character, since it doesn't do any real encoding
// conversion.
//
// std::filesystem::path already stores paths using each platform's
// native representation, so path.c_str() returns exactly the type these
// APIs want on every platform, with proper OS-level encoding conversion
// instead of a naive byte-widen.
inline std::filesystem::path to_native_path(const std::string& path) {
    return std::filesystem::path(path);
}
