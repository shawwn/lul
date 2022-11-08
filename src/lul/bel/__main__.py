import lul.bel
import sys

if args := sys.argv[1:]:
    for filename in args:
        lul.bel.load(filename, globals())
    lul.bel.interact(local=globals())
else:
    lul.bel.run("""
    (mac time body (let v (uvar 'v) (let w (uvar 'w) (list 'let v '(seconds) (list 'let w (cons 'do body) (list 'print (list '* '1000 (list '- '(seconds) v)) '" msec") w)))))

    (def wait (f) (let it (f) (if it it (wait f))))

    (mac dec (v (o n 1)) (list 'set v (list '- v n)))

    ;(set dec (list 'lit 'mac (eval "lambda v, n=1: list('set', v, list('-', v, n))" globe)))

    (def compose2 (f g) (macro args (list f (cons g args))))

    (def main ((o k 100))
      (do (set n k)
          (time (wait (fn () (do (dec n) '(print n) (id n 0)))))))
          
    (def main2 ((o n 100))
      (let self nil
        (def self (n)
          (if (<= n 0) 0 (self (- n 1))))
        (time (self n))))

    ;(main 1000)
    (main2 1000)

    (def main3 ((o n 100) (o it (where n)))
      (let self nil
        (def self ()
          ;(dec n)
          ;(set n (- n 1))
          ;(assign (where n) (- n 1))
          ;(assign (where n t) (- n 1))
          (assign it (- n 1))
          (if (<= n 0) 0 (self)))
        (dyn foo 42
          (time (self)))))
        
    ;(main3 1000)
    """, globals())

# lul.bel.readbel("(do (set n 10) (time (wait (fn () (do (dec n) (print n) (id n 0))))))")