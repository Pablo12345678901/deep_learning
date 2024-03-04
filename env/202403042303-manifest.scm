;; Steps to set the virtual environment with the Guix package manager:

;; 1. Run the below command :
;; Replace '$HOME' by your own absolute home path within the below command.
#!
    guix shell --container --network --preserve='^DISPLAY$' -m THIS-MANIFEST.scm --share=$HOME/.cache --share=/var/guix/daemon-socket/socket
!#
;; Detailed explanation of the above command :
;;    guix shell : create a shell
;;    --container : spawn a container isolated from the rest of the system. It does not know other than itself. Furthermore, unset all environment variables.
;;    --network : with a network access (container by default has not)
;;    -m : download, build and provides to the shell the packages defined in the manifest.
;;    --preserve : preserve an environment variable. Else would be unset by '--container'. Here, preserve the display to show things on screen.
;;    --share : give to the container an access to a directory
;;        Here gives access to :
;;        - the cache to download pip packages (else it shows a 'not enough space' ERROR)
;;        - the guix daemon socket to enable the download of packages from the shell. 

;; 2A. For Download the python environment with the pip requirements :
;; IMPORTANT : comment the lines of the guix packages that will be download through guix package manager.
;; (Refer to the list of Guix packages below)
;; Among others : comment the 'GDAL' and 'numpy' lines.
#!
    pip3 install --user -r GUIX-REQUIREMENTS-FILE.txt
!#

;; 3. Remove the pip packages that are not compatible with Guix :
;; (Refer to the list of Guix packages below)
#!
    pip3 uninstall numpy GDAL threadpoolctl
!#

;; 4. Re-install those packages with guix :
;; (Refer to the list of Guix packages below)
#!
    guix package -i gdal python-numpy python-threadpoolctl
!#

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; REMINDER ABOUT MANIFEST THEORY :
;; This "manifest" file can be passed to 'guix package -m' to reproduce
;; the content of your profile.  This is "symbolic": it only specifies
;; package names.  To reproduce the exact same profile, you also need to
;; capture the channels being used, as returned by "guix describe".
;; See the "Replicating Guix" section in the manual.

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(use-modules
 (gnu system) ;; for the %base-packages  
 (guix inferior) ;; for the channels use
 (guix channels) ;; for the channels use
 (srfi srfi-1) ;; SRFI is an acronym for Scheme Request For Implementation. The SRFI documents define a lot of syntactic and procedure extensions to standard Scheme as defined in R5RS.
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

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

  ;; The basis tools for development :
  ;; See the full list in GUIX-SOURCE-CODE/gnu/system.scm. Among others :
  ;;"bash"
  ;;"bash-completion" ;; for double tab to show possibilities
  ;;"coreutils" ;; for 'ls'
  ;;"findutils" ;; for 'find'
  ;;"grep"
  ;;"less"
  ;;"nano"
  (packages->manifest
   %base-packages)

  ;; Useful tools
  (specifications->manifest
   (list
    "guix" ;; To download guix packages.
    "nss-certs" ;; To enable git check of ssl certificates for the Guix daemon while using 'guix package ...'
    "xdg-utils" ;; To open files with the suitable tool depending on the MIME type.
  ))

  ;; Specific packages linked to the project
  (specifications->manifest
   (list
    ;; Python version
    "python" ;; v3.*
    "python:tk" ;; For 'tkinter' module within Python
    ;;; Specific Python packages for which the pip version does not works with Guix
    ;; So a Guix package build is required.
    "gdal" ;; For the osgeo module
    "python-numpy"
    "python-pip"
    "python-threadpoolctl"
  ))
    
#!
  ;; Specific packages from above custom channels 
  (packages->manifest
   (list
    ;;(first (lookup-inferior-packages inferior "PACKAGE-NAME")) ;; Syntax example.
  ))
!#

 )
)
