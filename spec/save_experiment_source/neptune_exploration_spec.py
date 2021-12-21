import neptune.new as neptune
from mamba import description, it
from src.utils.config_parser import config

with description("Neptune exploration", "exploration") as self:
    with it("Can connect to using the API-key"):
        project = "sjsivertandsanderkk/test-project"

        run = neptune.init(project=project)
        # Add tag for easy filtering later
        run["sys/tags"].add(["test"])
        # Save config
        run["options"] = config.dump()

        run.stop()
