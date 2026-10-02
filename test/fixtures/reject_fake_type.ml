type 'a gpu_array = Fake of 'a
let reject_fake_type (x : float gpu_array) = let _ = x in ()
