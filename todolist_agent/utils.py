class Spy:
    def __init__(self):
        self.called_tools = []

    def __call(self, run):
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )


def extract_tool_info(called_tools, tool_name):
    """Extract information from tool calls for both patches and new memories

    Args:
        called_tools: List of tool calls from the model
        tool_name: Name of the schema tool.
    """

    changes = []
    for call_group in called_tools:
        for call in call_group:
            if call["name"] == "PatchDoc":
                changes.append(
                    {
                        "type": "update",
                        "doc_id": call["args"]["json_doc_id"],
                        "planned_edits": call["args"]["planned_edits"],
                        "value": call["args"]["patches"][0]["value"],
                    }
                )
            elif call["name"] == tool_name:
                changes.append({"type": "new", "value": call["args"]})

        result_parts = []
        for change in changes:
            if change["type"] == "update":
                result_parts.append(
                    f"Document {change['doc_id']} updated:\n"
                    f"Plan: {change['planned_edits']}\n"
                    f"Added Content: {change['value']}"
                )
            elif change["type"] == "new":
                result_parts.append(
                    f"New {tool_name} created:\nContent: {change['value']}"
                )
        return "\n\n".join(result_parts)
