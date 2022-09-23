"""Tests for distutils.command.clean."""
import unittest

from test.support import run_unittest

from ... import bel


class BelTestCase(unittest.TestCase):

    def read(self, string, more=None):
        form, pos = bel.reader.read_from_string(string, mode="bel", more=more)
        return form

    def test_reader(self):
        read = self.read
        test = self.assertEqual
        test(bel.nil, read(""))
        #     (test= "nil" (read "nil"))
        test("nil", read("nil"))
        #     (test= 17 (read "17"))
        test("17", read("17"))
        #     (test= 0.015 (read "1.5e-2"))
        test("1.5e-2", read("1.5e-2"))
        #     (test= 15 (read "0xF"))
        test("0xF", read("0xF"))
        #     (test= -15 (read "-0Xf"))
        test("-0Xf", read("-0Xf"))
        #     (test= "0x" (read "0x"))
        test("0x", read("0x"))
        #     (test= "-0X" (read "-0X"))
        test("-0X", read("-0X"))
        #     (test= "-0Xg" (read "-0Xg"))
        test("-0Xg", read("-0Xg"))
        #     (test= true (read "true"))
        test("true", read("true"))
        #     (test= (not true) (read "false"))
        test("false", read("false"))
        #     (test= 'hi (read "hi"))
        test("hi", read("hi"))
        #     (test= '"hi" (read "\"hi\""))
        test('"hi"', read('"hi"'))
        #     (test= "|hi|" (read "|hi|"))
        test("|hi|", read("|hi|"))
        #     (test= '(1 2) (read "(1 2)"))
        test(["1", "2"], read("(1 2)"))
        #     (test= '(1 (a)) (read "(1 (a))"))
        test(["1", ["a"]], read("(1 (a))"))
        #     (test= '(quote a) (read "'a"))
        test(["quote", "a"], read("'a"))
        #     (test= '(quasiquote a) (read "`a"))
        test(["quasiquote", "a"], read("`a"))
        #     (test= '(quasiquote (unquote a)) (read "`,a"))
        test(["quasiquote", ["unquote", "a"]], read("`,a"))
        #     (test= '(quasiquote (unquote-splicing a)) (read "`,@a"))
        test(["quasiquote", ["unquote-splicing", "a"]], read("`,@a"))
        #     (test= 2 (# (read "(1 2 a: 7)")))
        #     (test= 7 (get (read "(1 2 a: 7)") 'a))
        #     (test= true (get (read "(:a)") 'a))
        #     (test= 1 (- -1))
        #     (test= "0?" (read "0?"))
        test("0?", read("0?"))
        #     (test= "0!" (read "0!"))
        test("0!", read("0!"))
        #     (test= "0." (read "0."))
        test("0.", read("0."))

    def test_read_more(self):
        read = self.read
        test = self.assertEqual
        #     (test= 17 (read "17" true))
        test("17", read("17", True))
        #     (let more ()
        more = object()
        #       (test= more (read "(open" more))
        test(more, read("(open", more))
        #       (test= more (read "\"unterminated " more))
        test(more, read('"unterminated ', more))
        #       (test= more (read "|identifier" more))
        # test(more, read('|identifier', more))
        #       (test= more (read "'(a b c" more))
        test(more, read("'(a b c", more))
        #       (test= more (read "`(a b c" more))
        test(more, read("`(a b c", more))
        #       (test= more (read "`(a b ,(z" more))
        test(more, read("`(a b ,(z", more))
        #       (test= more (read "`\"biz" more))
        test(more, read("`\"biz", more))
        #       (test= more (read "'\"boz" more)))
        test(more, read("'\"boz", more))
        #     (let ((ok e) (guard (read "(open")))
        #       (test= false ok)
        #       (test= "Expected ) at 5" (get e 'message)))
        test(more, read("(;foo", more))

    def test_simple_run(self):
        self.assertEqual(1, 1)
        self.assertEqual(1, bel.bel("1"))
        self.assertEqual(1, bel.bel("1"))

def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(BelTestCase)

if __name__ == "__main__":
    run_unittest(test_suite())
