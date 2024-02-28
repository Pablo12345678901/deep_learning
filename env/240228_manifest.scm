;; Use this file with the guix package manager :
;;    guix shell --container -m PATHTOTHISMANIFEST --network
;;
;; This "manifest" file can be passed to 'guix package -m' to reproduce
;; the content of your profile.  This is "symbolic": it only specifies
;; package names.  To reproduce the exact same profile, you also need to
;; capture the channels being used, as returned by "guix describe".
;; See the "Replicating Guix" section in the manual.

(use-modules
 (guix inferior)
 (guix channels)
 (srfi srfi-1)
)

;; My channels : content copied/pasted from the original file located ~/.config/guix/channels.scm
(define channels
 (cons*
;;#!
 ;; My custom channel
  (channel
   (name 'guix-custom-channel)
   (url "https://github.com/Pablo12345678901/guix-custom-channel.git")
   (introduction
    (make-channel-introduction "7e17f1a6211af7fa6096c2fb2372bf3347d34faa"
     (openpgp-fingerprint "0D7B 6A70 BC67 2797 FB0A 76C7 97A2 37BC 430D 199D")
    ) ;; First commit and public fingerprint
   )
  )
;;!#

#!
  ;; AVOID TO USE THE BELOW CHANNEL - ONLY IF NO OTHER SOLUTION AVAILABLE
  ;; Nonguix channel
  (channel
   (name 'nonguix)
   (url "https://gitlab.com/nonguix/nonguix")
   ;; Enable signature verification:
   (introduction
    (make-channel-introduction "897c1a470da759236cc11798f4e0a5f7d4d59fbc"
     (openpgp-fingerprint "2A39 3FFF 68F4 EF7A 3D29  12AF 6F51 20A0 22FB B2D5"))))
  ;; AVOID TO USE THE BELOW CHANNEL - ONLY IF NO OTHER SOLUTION AVAILABLE
!#
 
 %default-channels
))

;; When combined with channels (see Channels), inferiors provide a simple way to interact with a separate revision of Guix. 
(define inferior
  ;; An inferior representing the above revision.
  (inferior-for-channels channels))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Package list
(concatenate-manifests
 (list

  ;; Packages from above custom channels 
  (packages->manifest
   (list
    (first (lookup-inferior-packages inferior "my-micromamba"))
    ;;(first (lookup-inferior-packages inferior "my-python-gdal")) ;; Missing 
    (first (lookup-inferior-packages inferior "my-python-gensim"))
  ))

  ;; Other packages
  (specifications->manifest
   (list
    "bash"
    "bash-completion" ;; for double tab to show possibilities
    "coreutils" ;; for 'ls'
    "findutils" ;; for 'find'
    "gdal" ;; TEST 240228 for python osgeo module
    "grep"
    "guix"
    "less"
    "python"
    "python-geopandas"
    "python-keras"
    "python-numpy"
    "python-pip"
    ;;"python-protobuf"
    "python-scikit-learn"
    "python-scipy"
    "python-tqdm"
    "tensorflow"
  ))
 )
)
