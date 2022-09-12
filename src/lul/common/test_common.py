import unittest
from .. import common as lib


class CommonTest(unittest.TestCase):
    def assertCompiled(self, name: str, result: str, msg=None):
        id = lib.compile_id(name)
        self.assertEquals(id, result, msg)
        name2 = lib.uncompile_id(id)
        self.assertEquals(name, name2, msg="Roundtrip failed")

    def assertUncompiled(self, id: str, result: str, msg=None):
        self.assertEquals(lib.uncompile_id(id), result, msg)

    def check(self, x: str):
        y = lib.uncompile_id(lib.compile_id(x))
        self.assertEquals(x, y, f"Roundtrip failed for {x!r}. Compiled: {lib.compile_id(x)!r}")

    def test_common(self):
        augassign = [
            '+=',
            '-=',
            '*=',
            '@=',
            '/=',
            '%=',
            '&=',
            '|=',
            '^=',
            '<<=',
            '>>=',
            '**=',
            '//=']
        bitwise = "| ^ & >> <<".split()
        compare = "".split()
        infix = "- + / // * % @ **".split()
        for op in augassign + bitwise + compare + infix:
            self.check(op)

    def test_dunder(self):
        self.assertCompiled("--init--", "__init__")
        self.assertCompiled("--add--", "__add__")

    def test_private(self):
        self.assertCompiled("y--get", "y__get")
        self.assertCompiled("-get", "_get")

    def test_predicate(self):
        self.assertCompiled("list?", "listp")
        self.assertCompiled("foo-bar?", "foo_bar_p")

if __name__ == '__main__':
    unittest.main()