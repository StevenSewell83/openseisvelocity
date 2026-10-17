!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! PROGRAM resids
! Simple program which computes
! vpvs residuals
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PROGRAM resids
IMPLICIT NONE
INTEGER :: i,isum,istat,id1,id2
INTEGER, PARAMETER :: imax=100000000
REAL :: rms,var,rd1,rd2,av
CHARACTER(LEN=20):: otime,mtime
!
OPEN(UNIT=10,FILE='vpvsresids.in',STATUS='old')
READ(10,*)otime
READ(10,*)mtime
CLOSE(10)
OPEN(UNIT=20,FILE=otime,STATUS='old')
OPEN(UNIT=30,FILE=mtime,STATUS='old')
isum=0.0
rms=0.0
av=0.0
DO i=1,imax
   READ(20,*,END=101)id1,rd1
   READ(30,*,END=101)id2,rd2
   IF(id1.EQ.1.AND.id2.EQ.1)THEN
      isum=isum+1
      rms=rms+(rd1-rd2)**2
      av=av+(rd1-rd2)
   ENDIF
ENDDO
101 CONTINUE
CLOSE(20)
CLOSE(30)
rms=SQRT(rms/REAL(isum))
av=av/REAL(isum)
var=0.0
OPEN(UNIT=20,FILE=otime,STATUS='old')
OPEN(UNIT=30,FILE=mtime,STATUS='old')
DO i=1,imax
   READ(20,*,END=102)id1,rd1
   READ(30,*,END=102)id2,rd2
   IF(id1.EQ.1.AND.id2.EQ.1)THEN
      var=var+((rd1-rd2)-av)**2
   ENDIF
ENDDO
102 CONTINUE
var=var/REAL(isum)
WRITE(6,*)rms,var
CLOSE(20)
CLOSE(30)
END PROGRAM resids
