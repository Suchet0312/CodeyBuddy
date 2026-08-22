# from parser import parse_file
# content = parse_file(file_path="file")
def extractor(content, file_path):

    lines = content.splitlines()

    chunks = []
    current_chunk = []

    start_line = None
    function_name = None

    for line_number, line in enumerate(lines, start=1):

        if line.startswith("def "):

            # Save previous function
            if current_chunk:
                chunks.append({
                    "content": "\n".join(current_chunk),
                    "metadata": {
                        "file": file_path,
                        "function": function_name,
                        "start_line": start_line,
                        "end_line": line_number - 1,
                        "language": "python"
                    }
                })

            # Start new function
            current_chunk = [line]
            start_line = line_number
            function_name = line.split("(")[0].replace("def ", "").strip()

        else:

            if current_chunk:
                current_chunk.append(line)

    # Save last function
    if current_chunk:
        chunks.append({
            "content": "\n".join(current_chunk),
            "metadata": {
                "file": file_path,
                "function": function_name,
                "start_line": start_line,
                "end_line": len(lines),
                "language": "python"
            }
        })

    return chunks