from expects import equal, expect
from mamba import description, it

with description("Mamba test runner", "unit") as self:
    with it("Can run tests"):
        expect(True).to(equal(True))
