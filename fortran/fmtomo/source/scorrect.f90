program scorrect
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! This simple program corrects for
! source locations that lie outside
! the model region.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
implicit none
INTEGER :: i,j,k,ns
INTEGER :: nnr,nnt,nnp,ni,nip,nit
INTEGER, PARAMETER :: i10=SELECTED_REAL_KIND(10,100)
INTEGER, PARAMETER :: i5=SELECTED_REAL_KIND(5,10)
REAL(KIND=i10), PARAMETER :: pi=3.141592653589793
REAL(KIND=i5), DIMENSION(:), ALLOCATABLE :: sradr,slatr,slonr,stp
REAL(KIND=i5), DIMENSION(:), ALLOCATABLE :: covsr,covst,covsp
REAL(KIND=i10) :: gsrp,gstp,gspp,gorp,gotp,gopp,dc,gc
REAL(KIND=i10) :: dept,depb,latn,lats,lone,lonw,rearth
REAL(KIND=i10) :: deptn,depbn,rd1
INTEGER, DIMENSION(:), ALLOCATABLE :: lots,nps
INTEGER, DIMENSION(:,:,:), ALLOCATABLE :: paths,patht
INTEGER, DIMENSION(:,:), ALLOCATABLE :: nsdf
INTEGER, PARAMETER :: maxsd=50,maxs=60,maxp=40
CHARACTER (len=10), DIMENSION(:), ALLOCATABLE :: tpath
CHARACTER (len=15) :: sfile,pfile,ifile
!!!
open(unit=10,file='scorrect.in',status='old')
read(10,*)sfile
read(10,*)pfile
read(10,*)ifile
read(10,*)dc
read(10,*)gc
read(10,*)rearth
close(10)
OPEN(UNIT=10,FILE=sfile,STATUS='old')
READ(10,*)ns
ALLOCATE(sradr(ns),slatr(ns),slonr(ns))
ALLOCATE(covsr(ns),covst(ns),covsp(ns))
ALLOCATE(lots(ns),nps(ns))
ALLOCATE(tpath(ns))
ALLOCATE(nsdf(maxs,ns))
ALLOCATE(stp(ns))
ALLOCATE(paths(maxs,maxp,ns),patht(maxs,maxp,ns))
stp=0.0
DO i=1,ns
   READ(10,*)lots(i)
   IF(lots(i).EQ.1)READ(10,*)tpath(i)
   READ(10,*)sradr(i),slatr(i),slonr(i)
   READ(10,*)nps(i)
   DO j=1,nps(i)
      READ(10,*)nsdf(j,i)
      READ(10,*)paths(1:2*nsdf(j,i),j,i)
      READ(10,*)patht(1:nsdf(j,i),j,i)
   ENDDO
ENDDO
CLOSE(10)
open(unit=20,file=pfile,status='old')
read(20,*)nnr,nnt,nnp
read(20,*)gsrp,gstp,gspp
read(20,*)gorp,gotp,gopp
close(20)
!
! Compute bounds for depth, latitude and longitude
!
dept=-gorp
depb=gorp-(nnr-1)*gsrp
depb=-depb
lats=gotp
latn=gotp+(nnt-1)*gstp
lonw=gopp
lone=gopp+(nnp-1)*gspp
!
! Check interface levels
!
open(unit=30,file=ifile,status='old')
READ(30,*)ni
READ(30,*)nit,nip
READ(30,*)
READ(30,*)
DO i=1,ni
   DO j=1,nit
      DO k=1,nip
         IF(i.eq.1.and.j.eq.1.and.k.eq.1)THEN
            READ(30,*)deptn
         ELSE IF(i.EQ.ni.and.j.eq.1.and.k.eq.1)THEN
            READ(30,*)depbn
         ELSE
            READ(30,*)rd1
         ENDIF
      ENDDO
   ENDDO
ENDDO
depbn=rearth-depbn
deptn=rearth-deptn
IF(deptn.GT.dept)dept=deptn
IF(depbn.LT.depb)depb=depbn
!
! Now shrink bounds to ensure that no events lie on the boundary
!
dept=dept+dc
depb=depb-dc
lats=lats+gc
latn=latn-gc
lonw=lonw+gc
lone=lone-gc
!
! Now go through all source locations and check that they fall
! within bounding region
!
DO i=1,ns
   IF(lots(i).EQ.1)cycle
   IF(sradr(i).LT.dept)THEN
      sradr(i)=dept
      write(6,*)"Correcting depth of source ",i
   ENDIF
   IF(sradr(i).GT.depb)THEN
      sradr(i)=depb     
      write(6,*)"Correcting depth of source ",i
   ENDIF
   IF(slatr(i).LT.lats)THEN
      slatr(i)=lats
      write(6,*)"Correcting latitude of source ",i
   ENDIF
   IF(slatr(i).GT.latn)THEN
      slatr(i)=latn
      write(6,*)"Correcting latitude of source ",i
   ENDIF
   IF(slonr(i).LT.lonw)THEN
      slonr(i)=lonw
      write(6,*)"Correcting longitude of source ",i
   ENDIF
   IF(slonr(i).GT.lone)THEN
      slonr(i)=lone
      write(6,*)"Correcting longitude of source ",i
   ENDIF
ENDDO
!
! Now write results to file
!
OPEN(UNIT=10,FILE=sfile,STATUS='unknown')
WRITE(10,*)ns
DO i=1,ns
   WRITE(10,*)lots(i)
   IF(lots(i).EQ.1)WRITE(10,'(a10)')tpath(i)
   WRITE(10,*)sradr(i),slatr(i),slonr(i)
   WRITE(10,*)nps(i)
   DO j=1,nps(i)
      WRITE(10,*)nsdf(j,i)
      WRITE(10,*)paths(1:2*nsdf(j,i),j,i)
      WRITE(10,*)patht(1:nsdf(j,i),j,i)
   ENDDO
ENDDO
CLOSE(10)
DEALLOCATE(sradr,slatr,slonr)
DEALLOCATE(covsr,covst,covsp)
DEALLOCATE(lots,nps,tpath)
DEALLOCATE(nsdf,stp,paths,patht)
end program scorrect
