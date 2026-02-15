import Types

namespace Utils

def enumFilterMap (f : Nat -> α -> Option β) (l : List α) : List β :=
  (List.zip (List.range l.length) l).filterMap (λ (i, v) => f i v)
  
def dualize (l : List α) : List α := l ++ l.reverse

end Utils
