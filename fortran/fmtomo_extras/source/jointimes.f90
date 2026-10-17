!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! PROGRAM jointimes
! This program simply joins together
! two source-receiver assocaition files
! into one
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PROGRAM jointimes
IMPLICIT NONE
INTEGER :: i,j,k,l,id1,nrp,nrs,isw,nr,mxs
INTEGER, PARAMETER :: maxr=300, maxs=5000
INTEGER, DIMENSION(maxr,maxs) :: sid
INTEGER, DIMENSION(maxr) :: ns
REAL, DIMENSION(maxr,maxs) :: tt,tte,sdep,slat,slon
REAL, PARAMETER :: tol=1.0e-4
CHARACTER(LEN=30) :: srcp,srcs,receivp,receivs,srcb
!
! srcp = Source-receiver association file for P
! srcs = Source-receiver association file for S
! receivp = Receiver file for P
! receivs = Receiver file for S
! srcb = Source-receiver association file for both
!
OPEN(UNIT=10,FILE='jointimes.in',STATUS='old')
READ(10,1)srcp
READ(10,1)srcs
READ(10,1)receivp
READ(10,1)receivs
READ(10,1)srcb
1 FORMAT(a30)
CLOSE(10)
!
! Begin by reading in srcp and receivp
!
tt=-10.0
OPEN(UNIT=20,FILE=srcp, STATUS='old')
OPEN(UNIT=30,FILE=receivp,STATUS='old')
READ(20,*)nrp
mxs=0
READ(30,*)id1
DO i=1,nrp
   READ(20,*)id1,ns(i)
   DO j=1,ns(i)
      READ(20,*)sid(i,j),tt(i,j),tte(i,j)
      READ(30,*)sdep(i,j),slat(i,j),slon(i,j)
      READ(30,*)id1
      READ(30,*)id1
      READ(30,*)id1
      IF(sid(i,j).GT.mxs)mxs=sid(i,j)
   ENDDO
ENDDO
CLOSE(20)
CLOSE(30)
!
! Now read in srcs and apply "corrections" for source-receiver
! associations where appropriate.
!
OPEN(UNIT=40,FILE=srcs, STATUS='old')
OPEN(UNIT=50,FILE=receivs,STATUS='old')
READ(40,*)nrs
READ(50,*)id1
DO i=nrp+1,nrp+nrs
   READ(40,*)id1,ns(i)
   DO j=1,ns(i)
      READ(40,*)sid(i,j),tt(i,j),tte(i,j)
      READ(50,*)sdep(i,j),slat(i,j),slon(i,j)
      READ(50,*)id1
      READ(50,*)id1
      READ(50,*)id1
!
!     Now check to see whether current source id needs to be updated
!
      isw=0
      DO k=1,i
         IF(k.EQ.i)THEN
            id1=j-1
         ELSE
            id1=ns(k)
         ENDIF
         DO l=1,id1
            IF(ABS(sdep(i,j)-sdep(k,l)).LT.tol)THEN
               IF(ABS(slat(i,j)-slat(k,l)).LT.tol)THEN
                  IF(ABS(slon(i,j)-slon(k,l)).LT.tol)THEN
                     sid(i,j)=sid(k,l)
                     isw=1
                     EXIT
                  ENDIF
               ENDIF
            ENDIF
         ENDDO
         IF(isw.EQ.1)EXIT
      ENDDO
      IF(isw.EQ.0)THEN
         mxs=mxs+1
         sid(i,j)=mxs
      ENDIF
   ENDDO
ENDDO
CLOSE(40)
CLOSE(50)
nr=nrp+nrs
!
! Now export the resulting src file
!
OPEN(UNIT=60,FILE=srcb,STATUS='unknown')
WRITE(60,*)nr
DO i=1,nr
   WRITE(60,*)i,ns(i)
   DO j=1,ns(i)
      WRITE(60,*)sid(i,j),tt(i,j),tte(i,j)
   ENDDO
ENDDO
CLOSE(60)
END PROGRAM jointimes
