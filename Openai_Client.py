"""
Thin wrapper around OpenAI's /v1/responses endpoint.

We send:
  - a text instruction telling the model exactly what to produce
  - the uploaded image (as a base64 data URL) so the model can see the UI to replicate

We ask the model to reply with STRICT JSON matching our schema, which we then
parse into a ComponentBundle.
"""

import base64
import json
import os
from typing import Any

import httpx

from models import ComponentBundle

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are generating a React component from a reference image.

The first uploaded image is the exact design target and must be matched exactly.
Do not use generic form defaults, default product-form layouts, default button colors, default spacing, or default input styles.

Critical constraints:
- The first image is the only source of truth.
- Ignore all standard UI patterns and templates.
- If the reference image differs from a generic modern form, follow the image, not the convention.
- Do not invent a large centered layout if the reference image is compact.
- Do not enlarge the title, do not center everything, do not widen the panel, and do not use default button styling.
- Do not use a generic add-product form template.
- Match the uploaded image visually at a glance, including layout, spacing, text size, alignment, button color, and field proportions.

Exact visual matching rules:
- Match the card/background size, position, and tone exactly from the first image.
- Match the title size, weight, color, capitalization, and placement exactly.
- Match the labels and field placements exactly, including the precise horizontal and vertical positioning of every label relative to its field in the source image.
- The label alignment must match the uploaded image exactly. Do not assume a generic left/right rule; follow the exact position shown in the reference.
- Critical label rule: do not center the labels. If the uploaded image shows labels aligned to the left, keep them aligned to the left. Do not center label text on its own line. In the CSS, ALWAYS use `text-align: left;` for label elements. Never use `text-align: center;` on labels. The label must be explicitly set to left alignment in the CSS rule.
- Label CSS structure rule: create a specific CSS selector for labels (e.g., `label { }`, `.form-label { }`, or `form label { }`) that includes: display: block; text-align: left; and margin-bottom set to the EXACT vertical gap measured from the uploaded image. The margin-bottom on the label is the space between the label baseline and the input field top edge. Do NOT use generic 4px, 8px, 10px, or 12px unless the image shows that exact value. Measure the gap from the image and use that exact value.
- Match the textbox width, height, padding, border radius, fill color, and text vertical centering exactly. The input box must match the source reference in both size and fill color; do not use generic gray boxes or default heights.
- Critical textbox fill color rule: read the exact background color of the input textbox from the uploaded image and use that exact color value. Do not use generic colors like white, light gray, or default browser input color. Measure the RGB or hex value of the textbox background from the image and replicate it exactly in the CSS. If the image shows a slightly off-white, pale gray, light blue, or any other shade, use that exact shade.
- Input CSS structure rule: create a specific CSS selector for inputs (e.g., `input { }`, `.form-input { }`, or `form input { }`) that includes: height set to the EXACT measured pixel height from the image, padding set to the EXACT measured values from the image, background-color set to the EXACT color from the image (not generic #e1e1e1, #f5f5f5, or #ffffff), and width: 100% with box-sizing: border-box. Do NOT use generic 32px, 36px, or 40px for height unless the image shows that exact value. Do NOT use generic 8px or 10px for padding unless the image shows that exact value.
- Keep the textbox height compact and faithful to the source image. If the image shows a shorter field, do not use a tall default input height.
- Match the button size, width, height, radius, fill color, and position exactly. The button width must be copied from the source image, not widened to a standard default or a big modern pill.
- Critical button rule: do not enlarge the button. If the source image shows a smaller button, keep it smaller. Do not turn it into a large filled block or a wide modern CTA.
- Match the form spacing and row gaps exactly as shown in the source image.
- Match all text alignment exactly to the source image: labels, input text, validation text, and button text must follow the uploaded image’s exact alignment, no overlap.
- Match the warning/error text placement exactly under the relevant field.
- Match the exact color palette from the uploaded image, including background, title, field fill, button fill, text color, and validation color. Do not substitute generic theme colors.
- Keep all text readable and non-overlapping.
- Use exact field height values from the source image. Do not use generic default input heights (like 40px, 36px, 44px). If the reference is shorter or taller, preserve the exact height.
- Use exact padding inside textboxes: measure the padding from the image. Do not use generic padding values like 10px, 12px, 8px; use only what the image shows.
- Use exact label spacing and margin values from the image. Do not use generic label margins like 4px, 8px, 12px. Measure the exact gap between the label and the input field in the reference image and use that exact value.
- Use exact button width from the source image. Do not make the button wider than the reference just because it looks nicer.
- Label positioning rule: if the image shows labels above the input fields with a specific gap, use that exact gap. Do not assume a generic 4px or 8px margin-bottom. Measure the visual distance from the label baseline to the input top edge in the reference image and replicate it exactly.
- Forbidden generic values: never use default-like heights (40px, 36px, 44px, 48px), default-like padding (8px, 10px, 12px, 16px), or default-like margins (4px, 8px) unless the image itself clearly shows those exact values. Instead, infer the exact measurements from the reference screenshot.

Critical override:
- If the image is small and compact, do not generate a large full-screen layout.
- If the image title is medium-sized, do not scale it to a huge hero title.
- If the image uses a dark compact panel, do not generate a wide dark canvas with giant hero text.
- If the image uses a gray field fill, use that gray fill exactly.
- If the image uses a muted gray button, use that muted gray button exactly.
- If the image uses a red validation message, place it under the relevant field exactly and not somewhere else.
- If the image and generic UI conventions conflict, the image wins.

The screenshot must be treated as a pixel-level design spec.
The component must visually match the first uploaded image at a glance, including layout, spacing, text size, alignment, button color, and field proportions.

Requirements:
- Use plain CSS in a separate file imported into the component.
- Use a single default-exported React functional component.
- Keep the code clean and production-ready.
- Add brief JSDoc comments for sensible props.
- Generate a matching Jest + React Testing Library test file.
- Respond with ONLY a single JSON object matching exactly:
{
  "component_name": "PascalCaseName",
  "component_code": "<full file contents of the .jsx file>",
  "css_code": "<full file contents of the .css file>",
  "test_code": "<full file contents of the .test.jsx file>"
}
"""


class OpenAIError(RuntimeError):
    pass


def _image_bytes_to_data_url(image_bytes: bytes, content_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


async def generate_component_from_image(
    image_bytes: bytes,
    content_type: str,
    extra_instructions: str | None = None,
    desired_component_name: str | None = None,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    timeout: float = 90.0,
) -> tuple[ComponentBundle, dict[str, Any]]:
    """
    Calls OpenAI's /v1/responses endpoint with the given image and returns a
    parsed ComponentBundle plus the raw usage dict from the API response.

    If desired_component_name is given, the model is instructed to use that
    exact name, and the returned bundle's component_name is force-set to it
    afterward so the caller's naming is always honored regardless of what the
    model actually returns.
    """
    api_key = api_key or os.environ.get("THIRD_PARTY_API_KEY")
    if not api_key:
        raise OpenAIError("THIRD_PARTY_API_KEY is not set")

    data_url = _image_bytes_to_data_url(image_bytes, content_type)

    user_text = (
        "Generate the React component, CSS file, and unit test for the UI in this image. "
        "⚠️ ABSOLUTE REQUIREMENT: EVERY CSS value must come from the uploaded image. There are NO defaults, NO assumptions, and NO standard UI patterns. The uploaded image is the ONLY source of truth. "
        "MEASUREMENT PROTOCOL (MANDATORY): Before writing ANY CSS line, you MUST examine the uploaded image and physically measure these EXACT pixel values: "
        "(1) INPUT TEXTBOX HEIGHT: What is the exact pixel height of the input field from top border to bottom border? Example: if it is 28px tall, write 28px. If 30px, write 30px. Do NOT write 32px, 36px, 40px, or 44px unless the image clearly shows that exact value. "
        "(2) INPUT TEXTBOX PADDING: What is the exact padding inside the textbox (distance from border edge to where text starts)? Measure top and bottom separately. Example: if top is 5px and bottom is 5px, write padding: 5px 12px; Do NOT write 8px, 10px, or 12px unless the image shows that exact value. "
        "(3) LABEL-TO-INPUT SPACING: What is the exact vertical gap between the label text bottom and the input field top? Measure it in pixels. This is the margin-bottom of the label or margin-top of the input. Example: if the gap is 8px, use margin-bottom: 8px on the label. If it is 6px, use 6px. Do NOT assume 4px, 8px, 10px, or 12px unless the image shows that exact value. "
        "(4) INPUT TEXTBOX COLOR: What color are the textbox backgrounds? Read the RGB or hex value from the image. Do NOT use #ffffff (white), #f5f5f5 (light gray), #ecf0f1, #e1e1e1, or any standard palette color. Read the EXACT shade from the image. If the textbox is a slightly off-white cream, pale beige, very light gray, or light blue, measure that EXACT color. "
        "(5) LABEL STYLING: What size are the labels? What color? What weight? Measure from the image. Do NOT use generic 14px or 16px font size unless the image shows that exact size. "
        "(6) FORM CONTAINER: What is the exact background color of the form container? What is the exact width? What is the exact padding? Read from the image. "
        "This is a strict screenshot-matching task. The uploaded image is the exact design source and must be followed as closely as possible in layout, spacing, alignment, typography, colors, and control sizing. "
        "Do not use generic form defaults or compact UI patterns when the reference image differs. Instead, match the actual image geometry first and preserve any title, labels, field widths, row spacing, button placement, button sizing, warning box placement, exact colors, and text alignment exactly as shown. "
        "FORBIDDEN GENERIC VALUES: Never use standard form heights like 40px, 36px, 44px, 48px, 56px. Never use standard padding values like 8px, 10px, 12px, 16px unless you have measured and confirmed the image shows those exact values. Never use standard margin values like 4px, 8px, 12px unless you have measured and confirmed the image shows those exact values. Measure the exact height, padding, and margins from the reference image pixel-by-pixel and use only those exact values in your CSS. "
        "Textbox measurement rule: read the input field height from the image carefully. Measure the top and bottom edges of the input box. Measure the internal padding from the border edge to the text edge. Do not assume a standard 40px or 44px height; use the exact height shown in the reference. If the textbox is 28px tall, use 28px. If it is 32px, use 32px. Do not round or estimate. "
        "Label spacing rule: measure the vertical gap between the label text baseline and the input field top edge in the reference image. Use that exact gap value. Do not assume 4px, 8px, or 12px; measure what the image actually shows. If the gap is 6px, use 6px. If it is 10px, use 10px. If it is 2px, use 2px. "
        "Textbox vertical alignment is critical: the input text must be vertically centered inside each textbox, with the exact field height, padding, line-height, and font size copied from the reference image. Do not leave default browser text alignment or generic input padding that shifts the text upward/downward inside the field. "
        "Textbox fill color rule: before generating CSS, carefully read the exact background color of every textbox/input field from the uploaded image. Measure or infer the RGB hex value of the textbox fill. Do not use generic white (#ffffff), generic light gray (#f5f5f5, #ecf0f1), or default browser colors. Use the exact shade shown in the reference image. If the textbox is slightly off-white, cream, light beige, pale gray, light blue, or any other specific color, replicate that exact color in the CSS background-color property. "
        "Header safety rule: the heading must NEVER overlap, collapse, clip, or merge. Keep the headline readable on a single line unless the reference image clearly shows multiple lines. If the title is long, reduce font size or letter spacing; do not allow characters to touch or overlap. "
        "Follow a pixel-close design system: exact spacing values, exact font sizes, exact border radius, exact button dimensions, exact input height, exact padding, exact line-height, and exact form gap rhythm. Do not approximate when the screenshot shows a clear value. "
        "Exact typography rule: match the uploaded image’s exact font scale and weight. Do not default to standard form typography. The title should be much larger than a normal form heading, the labels should be readable but not tiny, and the button text should be centered and visually proportional to the button. "
        "Exact color rule: match the uploaded screenshot’s exact colors. Use the uploaded image as the source of truth for the panel background, title text, input fill, button color, and error text. Do not substitute generic palette values or a hardcoded blue, teal, cyan, or black unless the reference itself shows that exact shade. Copy the button color from the image, not from a default theme. "
        "Image analysis rule: before generating CSS, read the screenshot’s actual visual color relationships and contrast. Infer brightness, saturation, and contrast from the reference and reproduce that palette instead of defaulting to generic theme colors. "
        "Color safety requirement: the button must literally match the uploaded image’s hue, brightness, and contrast. If the image shows a muted gray, dark gray, blue, green, or another tone, use that exact tone. Do not use a standard button palette or an assumed teal tint. "
        "Exact palette rule: match the uploaded image’s exact color values for the page background, title, labels, input fill, button fill, warning text, and all other visible surfaces. Do not infer or substitute generic palette values; the uploaded image is the exact color reference. "
        "Deterministic checklist before code generation: read the panel background color, read the exact title scale and weight, read the exact textbox scale and border radius, read the exact field height and internal padding (not 10px unless the image shows 10px), read the exact label-to-input vertical gap (not 4px unless the image shows 4px), read the exact vertical centering of the text, read the exact button scale and color, read the label and text sizes, read the spacing rhythm between all controls, and verify that no text overlaps before finalizing the output. "
        "Label positioning and alignment measurement: carefully inspect the reference image and measure the exact horizontal alignment of labels relative to their input fields. Measure the exact vertical gap from the label baseline to the input field top edge. Do not assume a generic left-aligned 4px gap; use only what the visual reference clearly shows. Label text must align exactly as positioned in the uploaded image. "
        "Text safety rule: keep all text fully readable and non-overlapping. The title must not collide with itself or with adjacent elements. Labels and button text must stay clearly separated and centered within their controls. The input text must sit in the visual center of each field with no top/bottom drift. "
        "Exact sizing rule: match the reference’s actual header scale, input dimensions, internal padding, line-height, and button dimensions. Do not enlarge the header to a generic oversized wordmark or enlarge the button to a generic default size. Match the screenshot’s actual proportions exactly. "
        "Layout rule: the reference depends on a 2-column form with a 16px horizontal gap and 12px label spacing unless the image clearly differs. Match the mockup pixel-close. "
        "Exact mockup rule for this image: the main card color, title color, input fill, button color, title size, input proportions, textbox padding, field height, line-height, and button dimensions must all match the reference image exactly. The title should reflect the screenshot’s actual scale and remain readable, the labels should be dark text on the same baseline, the inputs should have the correct fill, border radius, and centered text alignment, and the submit button should match the screenshot’s exact tone and dimensions from the uploaded image. Any additional control or field in the mockup must also match its exact position, size, color, and spacing. "
        "Look-and-fill rule: the visual appearance of every control must match the uploaded reference, including fill colors, borders, radii, shadows (if any), text colors, background color, and the internal alignment of text inside the textbox. Do not standardize or genericize fields into different shades or styles. "
        "For any image you receive, read the structure first: title position, label alignment, field size and shape, internal padding, vertical text centering, spacing rhythm, warning placement, button style, exact button width/height, and exact color palette. Then reproduce that structure in React and CSS. "
        "This is dynamic and not limited to product-form screens. The uploaded UI may be a different form, panel, or layout. Follow the actual structure of the uploaded image exactly, regardless of whether it is a product form, employee form, card, or another screen. "
        "If the image uses a wider canvas, larger title, bigger inputs, different column proportions, a different background color, a different button color, or a warning box under a specific field, follow that exact arrangement. If the image needs the wrapper/card to be wider to preserve alignment, expand it to match the reference. "
        "Button sizing rule: do not enlarge the button to a generic oversized pill unless the image itself does. Match the real button dimensions and positioning from the reference, even if it is smaller or more compact than a default default. "
        "Title geometry rule: keep the header at the top-left with sufficient width and line-height, no squashed letters, no negative tracking, no clipped edges, and no text collision. The screenshot should be readable and stable. "
        "No generic default form behavior: ignore standard compact form assumptions such as small titles, short labels, low-density spacing, standard blue-gray buttons, or narrow cards. Use the actual screenshot’s geometry and styling, including the emergency safety rule that the header must not overlap. "
        "The input text alignment must mirror the uploaded reference exactly: set explicit padding, line-height, and font-size to center the text vertically; do not rely on generic default browser input styling. "
        "Use the screenshot as the final authority for all sizes, colors, spacing, and alignment. Match the exact palette and text positioning from the image instead of defaulting to a generic theme or a product-form assumption. "
        "FINAL VALIDATION CHECKLIST: Before generating the JSON output, verify EVERY single CSS value: "
        "(1) INPUT HEIGHT: Is every input height value exactly measured from the image? NOT 32px, 36px, 40px, 44px unless the image shows that. Do not use common form heights. "
        "(2) INPUT PADDING: Is every input padding value exactly measured? NOT 8px, 10px, 12px unless the image shows that. Measure the exact distance from border to text. "
        "(3) LABEL MARGIN-BOTTOM: Is every label margin-bottom value exactly measured from the image as the gap between label and input? NOT 4px, 8px, 10px, 12px unless the image shows that. Verify the visual gap matches the CSS value. "
        "(4) TEXTBOX COLOR: Is every textbox background-color exactly from the image? NOT #ffffff, #f5f5f5, #ecf0f1, #e1e1e1, or any palette color unless the image shows that. "
        "(5) LABEL ALIGNMENT: Do all label elements use `text-align: left;` UNLESS the image clearly shows center or right? Verify the CSS selector has this property. "
        "(6) FORM CONTAINER: Do the width, padding, and background-color match the visual reference? Measure from image, not guessed. "
        "(7) FONT SIZES: Are all font sizes (title, labels, inputs) measured from the image? NOT standard 14px, 16px, 24px unless the image shows that. "
        "If ANY validation fails (e.g., input is 32px but image shows 30px, or label margin is 10px but image shows 6px, or textbox color is #e1e1e1 but image shows a different shade), STOP and correct the CSS immediately before outputting the JSON. Do NOT output incorrect values."
    )
    if desired_component_name:
        user_text += (
            f" Name the component exactly '{desired_component_name}' "
            f"(component_name, file names, exports, and imports must all use this name)."
        )
    if extra_instructions:
        user_text += f"\n\nAdditional instructions: {extra_instructions}"

    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        "text": {"format": {"type": "json_object"}},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(OPENAI_RESPONSES_URL, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        raise OpenAIError(f"OpenAI request failed: {exc}") from exc

    if resp.status_code >= 400:
        raise OpenAIError(f"OpenAI API error {resp.status_code}: {resp.text}")

    try:
        data = resp.json()
    except ValueError as exc:
        raise OpenAIError(f"Could not decode OpenAI JSON response: {resp.text[:1000]}") from exc

    output_text = _extract_output_text(data)

    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        cleaned = _coerce_json_object(output_text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise OpenAIError(
                f"Could not parse model output as JSON: {exc}\nRaw: {output_text[:2000]}"
            ) from exc

    try:
        bundle = ComponentBundle(**parsed)
    except (TypeError, ValueError) as exc:
        raise OpenAIError(f"Could not parse model output as ComponentBundle: {exc}\nRaw: {output_text[:2000]}") from exc

    if desired_component_name and bundle.component_name != desired_component_name:
        bundle.component_name = desired_component_name

    return bundle, data.get("usage", {})


def _extract_output_text(data: dict[str, Any]) -> str:
    """Extract the text payload from a /v1/responses response.

    OpenAI responses can return either a plain output_text field, nested content items,
    or a chat-completion-like choices payload depending on the model/path used.
    """
    if data.get("error"):
        raise OpenAIError(f"OpenAI returned an error payload: {data['error']}")

    if "output_text" in data and data["output_text"]:
        return data["output_text"]

    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in ("output_text", "text"):
                text = content.get("text")
                if text:
                    return text

    for choice in data.get("choices", []):
        message = choice.get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if text:
                        return text

    raise OpenAIError(f"No output_text found in OpenAI response: {json.dumps(data)[:1000]}")


def _coerce_json_object(raw_text: str) -> str:
    """Strip markdown fences and return the first JSON object payload in the text."""
    text = raw_text.strip()
    if not text:
        return text

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("```json", "", 1).replace("```JSON", "", 1)
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text