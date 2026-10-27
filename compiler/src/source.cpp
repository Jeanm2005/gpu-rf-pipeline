#include "sigflow/source.hpp"

#include <fstream>
#include <sstream>

namespace sigflow {

SourceFile::SourceFile(std::string display_path, std::string text)
    : path_(std::move(display_path)), text_(std::move(text))
{
    std::string current;
    for (char c : text_) {
        if (c == '\n') {
            if (!current.empty() && current.back() == '\r') {
                current.pop_back();
            }
            lines_.push_back(std::move(current));
            current.clear();
        } else {
            current += c;
        }
    }
    if (!current.empty() && current.back() == '\r') {
        current.pop_back();
    }
    lines_.push_back(std::move(current));
}

const std::string& SourceFile::line_text(int line) const
{
    static const std::string empty;
    if (line < 1 || static_cast<std::size_t>(line) > lines_.size()) {
        return empty;
    }
    return lines_[static_cast<std::size_t>(line - 1)];
}

bool read_file(const std::string& path, std::string& text)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    text = buffer.str();
    return true;
}

}  // namespace sigflow
