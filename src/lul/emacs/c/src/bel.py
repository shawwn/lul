from .lisp import *
from .fns import *
from .eval import *
from .data import *

class bel:
    @staticmethod
    def no(x):
        return NILP(x)

    # (def keep (f xs)
    #   (if (no xs)      nil
    #       (f (car xs)) (cons (car xs) (keep f (cdr xs)))
    #                    (keep f (cdr xs))))
    @staticmethod
    def keep(f, xs):
        if bel.no(xs):
            return Q.nil
        elif not bel.no(f(F.car(xs))):
            return F.cons(F.car(xs), bel.keep(f, F.cdr(xs)))
        else:
            return bel.keep(f, F.cdr(xs))

    # (def rem (x ys (o f =))
    #   (keep [no (f _ x)] ys))
    @staticmethod
    def rem(x, ys, f=Q.nil):
        if NILP(f):
            f = F.eq
        return bel.keep(lambda _: bel.no(f(_, x)), ys)

    @staticmethod
    def mem(elt: Lisp_Object, list: Lisp_Object, test: Lisp_Object = Q.nil):
        if NILP(test):
            test = Q.eq
        CHECK_CONS(list)
        # Lisp_Object tail = list;
        # FOR_EACH_TAIL (tail)
        for tail, _ in FOR_EACH_TAIL(list):
            # if (EQ (XCAR (tail), elt))
            #   return tail;
            if not NILP(F.funcall(test, XCAR(tail), elt)):
                return tail
        # CHECK_LIST_END (tail, list);
        # return Qnil;
        return Q.nil


    # (def dups (xs (o f =))
    #   (if (no xs)                   nil
    #       (mem (car xs) (cdr xs) f) (cons (car xs)
    #                                       (dups (rem (car xs) (cdr xs) f) f))
    #                                 (dups (cdr xs) f)))
    # @DEFUN("dups", S.dups, 1, 2, 0)
    @staticmethod
    def dups(xs: Lisp_Object, f: Lisp_Object = Q.nil):
        if NILP(xs):
            return Q.nil
        elif not NILP(bel.mem(F.car(xs), F.cdr(xs), f)):
            return F.cons(F.car(xs), bel.dups(bel.rem(F.car(xs), F.cdr(xs), f), f))
        else:
            return bel.dups(F.cdr(xs), f)
