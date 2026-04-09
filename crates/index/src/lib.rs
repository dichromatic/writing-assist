use writing_assist_core::SpanType;

pub fn supported_span_types() -> [SpanType; 5] {
    [
        SpanType::Heading,
        SpanType::Paragraph,
        SpanType::Section,
        SpanType::Window,
        SpanType::Scene,
    ]
}
