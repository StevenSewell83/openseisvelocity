!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! PROGRAM MKVPVS
! Simple program for making
! vp/vs parameter file
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PROGRAM mkvpvs
IMPLICIT NONE
INTEGER i,j,k,ng,ngt,nnr,nnlat,nnlon,nrlim,isum
REAL :: vpvs
REAL, PARAMETER :: rearth=6371.0
REAL*8 :: dnr,dnlat,dnlon,vp,vs,rd1
REAL*8 :: gor,golat,golon,vlim
CHARACTER(LEN=20) :: vgs,vgp,vg,vgo
OPEN(UNIT=10,FILE='mkvpvs.in',STATUS='old')
READ(10,*)vgp
READ(10,*)vgs
READ(10,*)vg
READ(10,*)vgo
READ(10,*)vlim
CLOSE(10)
!
! Read in reference Vp and Vs grids and compute
! vpvs ratio. Also output predicted Vp/Vs
!
OPEN(UNIT=10,FILE=vgp,STATUS='old')
OPEN(UNIT=20,FILE=vgs,STATUS='old')
OPEN(UNIT=30,FILE=vgo,STATUS='unknown')
READ(10,*)ng,ngt
READ(10,*)nnr,nnlat,nnlon
READ(10,*)dnr,dnlat,dnlon
READ(10,*)gor,golat,golon
READ(20,*)
READ(20,*)
READ(20,*)
READ(20,*)
WRITE(30,*)ng,ngt
WRITE(30,*)nnr,nnlat,nnlon
WRITE(30,*)dnr,dnlat,dnlon
WRITE(30,*)gor,golat,golon
rd1=0.
nrlim=(rearth-gor-vlim)/dnr+1
isum=0
DO i=1,nnr
   DO j=1,nnlat
      DO k=1,nnlon
         READ(10,*)vp
         READ(20,*)vs
         WRITE(30,*)vp/vs
         IF(i.GT.nrlim.AND.j.GT.1.AND.K.GT.1)THEN
            IF(i.LT.nnr.AND.j.LT.nnlat.AND.k.LT.nnlon)THEN
               rd1=rd1+vp/vs
               isum=isum+1
            ENDIF
         ENDIF
      ENDDO
   ENDDO
ENDDO
CLOSE(10)
CLOSE(20)
CLOSE(30)
write(6,*)vp/vs
rd1=rd1/REAL(isum)
vpvs=rd1
write(6,*)rd1
!
! Write output to file
!
OPEN(UNIT=30,FILE=vg,STATUS='unknown')
WRITE(30,*)ng,ngt
WRITE(30,*)nnr,nnlat,nnlon
WRITE(30,*)dnr,dnlat,dnlon
WRITE(30,*)gor,golat,golon
DO i=1,nnr
   DO j=1,nnlat
      DO k=1,nnlon
         WRITE(30,*)vpvs
      ENDDO
   ENDDO
ENDDO
CLOSE(30)
END PROGRAM mkvpvs
