#pragma once

// SigFlow value types (compiler/README.md, "Types").

#include <string>

namespace sigflow {

enum class ElementType { Cf32, F32 };

struct Type {
    enum class Kind {
        Stream,  // stream<cf32>: continuous samples at `rate`
        Frames,  // frames<elem, frame_size>: blocks of frame_size samples
    };
    Kind kind = Kind::Stream;
    ElementType element = ElementType::Cf32;
    int frame_size = 0;  // Frames only
    // Sample rate in samples per second, and how the user spelled it ("2.4e6"), so diagnostics
    // and dumps quote the source. Frames keep the rate of the stream they were cut from.
    double rate = 0.0;
    std::string rate_text;

    static Type stream(double rate, std::string rate_text)
    {
        Type t;
        t.kind = Kind::Stream;
        t.rate = rate;
        t.rate_text = std::move(rate_text);
        return t;
    }
    static Type frames(ElementType element, int frame_size, double rate, std::string rate_text)
    {
        Type t;
        t.kind = Kind::Frames;
        t.element = element;
        t.frame_size = frame_size;
        t.rate = rate;
        t.rate_text = std::move(rate_text);
        return t;
    }
};

// "stream<cf32>", "frames<f32, 1024>". The rate is not part of the name.
std::string type_name(const Type& type);

// The file extension rfgpu reads and writes for a value of this type: ".cf32" or ".f32".
const char* file_extension(const Type& type);

}  // namespace sigflow
