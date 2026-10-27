#pragma once

// A SigFlow source file held in memory, split into lines once so that every pass can attach the
// offending source line to a diagnostic without re-reading the file.

#include <string>
#include <vector>

namespace sigflow {

class SourceFile {
public:
    // `display_path` is what diagnostics print (e.g. "examples/two_channel.sf"); `text` is the
    // whole file. "\r\n" line endings are accepted and treated as "\n".
    SourceFile(std::string display_path, std::string text);

    const std::string& path() const { return path_; }
    const std::string& text() const { return text_; }

    // The text of 1-based line `line`, without its line terminator. Empty for a line past the
    // end of the file (e.g. the position of EOF in an empty file).
    const std::string& line_text(int line) const;

private:
    std::string path_;
    std::string text_;
    std::vector<std::string> lines_;
};

// Reads `path` into `text`. Returns false if the file cannot be opened.
bool read_file(const std::string& path, std::string& text);

}  // namespace sigflow
