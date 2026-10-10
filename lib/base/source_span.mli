(** Source locations for diagnostics. *)

type t = {
  file : string;
  start_line : int;
  start_col : int;
  end_line : int;
  end_col : int;
}

val synthetic : t
val to_string : t -> string

exception Error of t * string
val fail : t -> string -> 'a
