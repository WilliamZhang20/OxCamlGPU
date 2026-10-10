type t = { file : string; start_line : int; start_col : int; end_line : int; end_col : int }
let synthetic = { file="<generated>"; start_line=0; start_col=0; end_line=0; end_col=0 }
let to_string s = Printf.sprintf "%s:%d:%d" s.file s.start_line (s.start_col + 1)
exception Error of t * string
let fail loc message = raise (Error (loc, message))
