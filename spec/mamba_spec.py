from expects import equal, expect
from mamba import context, description, it

with description("Mamba test runner") as self:
    with it("starts with 0 - 0 score"):
        rafa_nadal = "Rafa Nadal"
        roger_federer = "Roger Federer"

        expect(True).to(equal(True))
