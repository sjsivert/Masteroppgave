from expects import equal, expect
from mamba import description, it

with description("Mamba test runner", "unit") as self:
    with it("starts with 0 - 0 score"):
        rafa_nadal = "Rafa Nadal"
        roger_federer = "Roger Federer"

        expect(True).to(equal(True))
