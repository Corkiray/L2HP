(define
   (problem bloques_problem)
   (:domain bloques)

   (:objects 
      a - block
      b - block
      c - block
   )

   (:init
      (on-table a)
      (clear a)
      (on-table b)
      (clear b)
      (on-table c)
      (clear c)
      (arm-empty )
   )

   (:goal
      (and 
         (on a b) 
         (on b c) 
         (clear a) 
      )
   )

)