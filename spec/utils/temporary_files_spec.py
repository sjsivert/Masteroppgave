import os

from expects import be_false, be_true, expect
from mamba import description, included_context, it, shared_context

from src.utils.temporary_files import temp_files

temp_path = "temp_path"
with shared_context("create temp file folder and remov again") as self:
    with it("creates temporary folder and removes it after"):
        temp_path = "temp_path"
        with temp_files(temp_path):
            expect(os.path.exists(temp_path)).to(be_true)

        expect(os.path.exists(temp_path)).to(be_false)

with description(temp_files, "unit"):
    with included_context("create temp file folder and remov again"):
        pass

    with it("do still run even if folder exists"):
        os.mkdir(temp_path)
        with included_context("create temp file folder and remov again"):
            pass
        expect(os.path.exists(temp_path)).to(be_true)

        os.rmdir(temp_path)
        expect(os.path.exists(temp_path)).to(be_false)
