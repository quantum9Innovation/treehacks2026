namespace Utils

def enumFilterMap (f : Nat -> α -> Option β) (l : List α) : List β :=
  (List.zip (List.range l.length) l).filterMap (fun (i, v) => f i v)
  
end Utils
