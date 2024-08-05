from textual.suggester import Suggester


class ColumnNameSuggestor(Suggester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, case_sensitive=True, **kwargs)
        self.columns = tuple[str, ...]()

    async def get_suggestion(self, value: str) -> str | None:
        if not value:
            return None
        tokens = value.rsplit(maxsplit=3)
        last_token = tokens[-1]
        if len(tokens) > 1:
            combos = ("and", "or", "AND", "OR")
            for misc in combos:
                if misc.startswith(last_token) and tokens[-2] not in combos:
                    return f"{value}{misc.removeprefix(last_token)}"

        for col in self.columns:
            if col.startswith(last_token):
                return f"{value}{col.removeprefix(last_token)}"

        return None
