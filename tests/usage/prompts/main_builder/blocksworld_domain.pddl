(define (domain bloques)
   (:requirements
      :strips :typing)

   (:types 
      block
   )

   (:predicates 
      (on-table ?b - block) ;  'block is resting on the table'
      (clear ?b - block) ;  'block has nothing on top'
      (on ?b1 - block ?b2 - block) ;  'block b1 is directly on top of block b2'
      (holding ?b - block) ;  'the robot is holding block b'
      (arm-empty ) ;  'the robot’s arm is empty'
   )

   (:action pickup
      :parameters (
         ?b - block
      )
      :precondition
         (and
             (on-table ?b) ; block must be on the table
             (clear ?b) ; nothing is on top of the block
             (arm-empty) ; arm must be free to pick up a block
         )
      :effect
         (and
             (not (on-table ?b)) ; block is no longer on the table
             (not (clear ?b)) ; block is no longer clear as it's held
             (not (arm-empty)) ; arm is no longer empty
             (holding ?b) ; robot is now holding the block
         )
   )

   (:action putdown
      :parameters (
         ?b - block
      )
      :precondition
         (and
             (holding ?b) ; robot must be holding the block
         )
      :effect
         (and
             (on-table ?b) ; block is placed on the table
             (clear ?b) ; block becomes clear
             (arm-empty) ; arm becomes empty
             (not (holding ?b)) ; robot is no longer holding the block
         )
   )

   (:action stack
      :parameters (
         ?b1 - block
         ?b2 - block
      )
      :precondition
         (and
             (holding ?b1) ; robot must be holding the top block
             (clear ?b2) ; bottom block must be clear
         )
      :effect
         (and
             (on ?b1 ?b2) ; b1 is on top of b2
             (clear ?b1) ; top block becomes clear
             (arm-empty) ; robot is no longer holding a block
             (not (clear ?b2)) ; bottom block is no longer clear
             (not (holding ?b1)) ; robot is no longer holding b1
         )
   )

   (:action unstack
      :parameters (
         ?b1 - block
         ?b2 - block
      )
      :precondition
         (and
             (on ?b1 ?b2) ; b1 is on top of b2
             (clear ?b1) ; b1 must be clear
             (arm-empty) ; arm must be empty
         )
      :effect
         (and
             (holding ?b1) ; robot is now holding b1
             (clear ?b2) ; b2 becomes clear
             (not (on ?b1 ?b2)) ; b1 is no longer on b2
             (not (clear ?b1)) ; b1 is no longer clear
             (not (arm-empty)) ; robot arm is no longer empty
         )
   )
)