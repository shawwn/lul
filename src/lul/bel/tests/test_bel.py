"""Tests for distutils.command.clean."""
import unittest

from test.support import run_unittest

from ... import bel


class BelTestCase(unittest.TestCase):

    def read(self, string, more=None):
        form, pos = bel.reader.read_from_string(string, mode="bel", more=more)
        return form

    def test_list(self):
        list = bel.list
        test_eq = self.assertEqual
        test_not = self.assertNotEqual
        test_is = self.assertIs
        test_lt = self.assertLess
        test_gt = self.assertGreater
        test_eq(list(1, 2, 3), list(1, 2, 3))
        test_not(list(1, 2, 4), list(1, 2, 3))
        mark = list("%mark")
        test_is(mark, mark)
        test_eq(42, {mark: 42}.get(mark))
        test_lt(list(1, 2), list(2, 2))
        test_lt(list(2, 1), list(2, 2))
        test_gt(list(2, 2), list(1, 2))
        test_gt(list(2, 2), list(2, 1))

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

    def test_null(self):
        test = self.assertIs
        # (define-test nil?
        #   (test= true (nil? nil))
        test(True, bel.null(bel.nil))
        #   (test= true (nil? null))
        #   (test= false (nil? true))
        test(False, bel.null(True))
        #   (test= false (nil? false))
        test(False, bel.null(False))
        #   (test= false (nil? (obj))))
        test(False, bel.null(object()))
        test(False, bel.null(0))

    def test_ok(self):
        test = self.assertIs
        # (define-test is?
        #   (test= false (is? nil))
        test(False, bel.ok(bel.nil))
        #   (test= false (is? null))
        #   (test= true (is? true))
        test(True, bel.ok(True))
        #   (test= true (is? false))
        test(True, bel.ok(False))
        #   (test= true (is? (obj))))
        test(True, bel.ok(object()))

    def test_no(self):
        test = self.assertIs
        # (define-test no
        #   (test= true (no nil))
        test(True, bel.no(bel.nil))
        #   (test= true (no null))
        #   (test= false (no true))
        test(False, bel.no(True))
        #   (test= true (no false))
        test(True, bel.no(False))
        #   (test= false (no (obj)))
        test(False, bel.no(object()))
        #   (test= false (no 0)))
        test(False, bel.no(0))

    def test_bel(self):
        self.assertEqual(1, 1)
        self.assertEqual(1, bel.bel("1"))

def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(BelTestCase)

if __name__ == "__main__":
    run_unittest(test_suite())
