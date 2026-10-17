!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! GLOBAL MODULE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODULE globalp
IMPLICIT NONE
REAL, DIMENSION(:,:,:), ALLOCATABLE :: dvpvs,vp,vs,dvs
REAL, DIMENSION(4) :: ui,vi,wi,uio,vio,wio
!
! vp = Reference P-velocity grid
! dvpvs = Differential VpVs ratio
! vs = Reference S-velocity grid
! dvs = S-velocity perturbation
! ui,vi,zi = B-spline basis values
! uio,vio,zio = Previous ui,vi and zi
!
END MODULE globalp

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! PROGRAM frvpvs
! This program is designed to compute
! Frechet derivatives needed for the
! inversion for Vp/Vs ratio. It also
! computes the observed and mode
! data
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PROGRAM frvpvs
USE globalp
IMPLICIT NONE
INTEGER :: i,j,k,l,m,n,ii,isum
INTEGER :: nvr,nvx,nvz,nrp,nrs,nr,nmp
INTEGER :: id1,id2,id3,id4
INTEGER :: npt,ivr,ivx,ivz,ivro,ivxo,ivzo
INTEGER :: nhp,ivrt,ivxt,ivzt,iddat
INTEGER, PARAMETER :: maxr=300,maxs=5000
INTEGER, DIMENSION(maxr) :: nslr,idr
INTEGER, DIMENSION(maxr,maxs) :: idtto,spas
INTEGER, DIMENSION(4) :: chp
REAL :: dvr,dvx,dvz,tobs,plt
REAL :: gor,gox,goz,vpvsr
REAL :: rd1,rd2,rd3
REAL :: rgr,rgx,rgz,rgro,rgxo,rgzo
REAL :: ri,xi,zi,drr,drx,drz,u,v,w
REAL :: vel,velps,velo,velpso
REAL :: vels,velso,velds,veldso
REAL :: dinc,dpl,mdat,rigr,rigx,rigz
REAL, PARAMETER :: pi=3.14159265359, rearth=6371.0
REAL, PARAMETER :: ftol=1.0e-5
REAL, DIMENSION(maxr,maxs) :: tto
REAL, DIMENSION(4) :: vrat
REAL, DIMENSION (:,:,:), ALLOCATABLE :: fdm
CHARACTER (LEN=20) :: rayf,vgf,vgrf,vgpf,srcf,frdf,vgsrf,vgsf
CHARACTER (LEN=20) :: otsf,otpf,otf,mtf,recp,recs,sarf
!
! rayf = Raypath file
! vgf = Vp/Vs grid file
! vgrf = Vp/Vs reference grid file
! vgpf = Vp grid file
! vgsrf = Vs reference grid file
! vgsf = Vs current grid file
! srcf = Source-receiver reference file
! otsf = Observed times for S
! otpf = Observed times for P
! otf = Observed data
! mtf = Model data
! nvr,nvx,nvz = Number of nodes in r,theta,phi
! dvr,dvx,dvz = Node separaton in r,theta,phi
! gor,gox,goz = Grid origin in r,theta,phi
! recp, recs = Receivers for P and S
! nrp,nrs = Number of receivers for P and S
! nr = Total number of receivers
! maxr = Maximum number of receivers
! maxs = maximum number of sources
! tto = Observed traveltimes
! nslr = Number of sources for receiver i
! idtto = Source-receiver ID for tto
! idr = Receiver pointer for receier i
! nmp = Number of matching pairs
! tobs = Observed data
! frdf = Frechet derivative file
! npt = Number of points in ray segment
! rgr,rgx,rgz = Ray path point
! rgro,rgxo,rgzo = Previous ray path point
! ivr,ivx,ivz = Cell location of ray path point
! ivro,ivxo,ivzo = Previous location of ray path point
! nhp = Number of ray segment-B-spline cell hit points
! ri,xi,zi = Edge of model coordinates
! vrat = length ratio of ray sub-segment
! chp = pointer to incremental change in r,x or z cell
! drr,drx,drz = Ray path point displacement in cell
! u,v,w = Ray path point proportional cell location
! ivrt,ivxt,ivzt = temporary ivr,ivx,ivz values
! vel = Current P velocity
! velps = Current d(Vp/Vs)
! vels = Current S velocity
! velds = current d(Vs)
! velo,velpso = Old vel and velps
! velso,veldso = Old vels and velds
! dinc = Incremental path length
! dpl = Length of segment between two ray points
! fdm = Frechet derivative matrix
! mdat = Model data prediction
! iddat = ID of model data (0=doesn't exist; 1=exists)
! rigr,rigx,rigz = end point of sub-segment of ray path
! ftol = tolerance for partial derivative
! sarf = File containing total number of sources and receivers
! spas = S-P time association
!
OPEN(UNIT=10,FILE='frvpvs.in',STATUS='old')
READ(10,*)rayf
READ(10,*)vgf
READ(10,*)vgrf
READ(10,*)vgpf
READ(10,*)vgsrf
READ(10,*)vgsf
READ(10,*)srcf
READ(10,*)otsf
READ(10,*)otpf
READ(10,*)otf
READ(10,*)mtf
READ(10,*)recp
READ(10,*)recs
READ(10,*)frdf
READ(10,*)sarf
CLOSE(10)
!
! Now read in differential Vp/Vs, Vp, Vs and reference Vs
!
OPEN(UNIT=10,FILE=vgf,STATUS='old')
OPEN(UNIT=20,FILE=vgrf,STATUS='old')
OPEN(UNIT=30,FILE=vgpf,STATUS='old')
OPEN(UNIT=40,FILE=vgsrf,STATUS='old')
OPEN(UNIT=50,FILE=vgsf,STATUS='old')
READ(10,*)
READ(10,*)nvr,nvx,nvz
nvr=nvr-2
nvx=nvx-2
nvz=nvz-2
ALLOCATE(dvpvs(0:nvr+1,0:nvx+1,0:nvz+1))
ALLOCATE(vp(0:nvr+1,0:nvx+1,0:nvz+1))
ALLOCATE(vs(0:nvr+1,0:nvx+1,0:nvz+1))
ALLOCATE(dvs(0:nvr+1,0:nvx+1,0:nvz+1))
READ(10,*)dvr,dvx,dvz
READ(10,*)gor,gox,goz
gox=gox+dvx*nvx
gox=pi/2.0-gox
gor=gor-rearth
gor=gor+dvr*nvr
goz=goz+dvz
DO i=1,4
   READ(20,*)
   READ(30,*)
   READ(40,*)
   READ(50,*)
ENDDO
DO i=0,nvr+1
   DO j=0,nvx+1
      DO k=0,nvz+1
         READ(10,*)dvpvs(nvr+1-i,nvx+1-j,k)
         READ(20,*)rd1
         dvpvs(nvr+1-i,nvx+1-j,k)=dvpvs(nvr+1-i,nvx+1-j,k)-rd1
         READ(30,*)vp(nvr+1-i,nvx+1-j,k)
         READ(40,*)vs(nvr+1-i,nvx+1-j,k)
         READ(50,*)dvs(nvr+1-i,nvx+1-j,k)
         dvs(nvr+1-i,nvx+1-j,k)=dvs(nvr+1-i,nvx+1-j,k)-vs(nvr+1-i,nvx+1-j,k)
      ENDDO
   ENDDO
ENDDO
!
! Convert to degrees
!
gox=gox*180./pi
goz=goz*180./pi
dvx=dvx*180./pi
dvz=dvz*180./pi
vpvsr=rd1
CLOSE(10)
CLOSE(20)
CLOSE(30)
CLOSE(40)
CLOSE(50)
!
! Determine number of unique receivers
!
OPEN(UNIT=20,FILE=recp,STATUS='old')
READ(20,*)nrp
DO i=1,nrp
   READ(20,*)
   READ(20,*)rd1,rd2,rd3,idr(i)
   READ(20,*)
   READ(20,*)
   READ(20,*)
   READ(20,*)
ENDDO
CLOSE(20)
OPEN(UNIT=30,FILE=recs,STATUS='old')
READ(30,*)nrs
DO i=nrp+1,nrp+nrs
   READ(30,*)
   READ(30,*)rd1,rd2,rd3,idr(i)
   READ(30,*)
   READ(30,*)
   READ(30,*)
   READ(30,*)
ENDDO
CLOSE(30)
!
! Now read in S-times and P-times
! and extract matching pairs
!
tto=-100.
OPEN(UNIT=40,FILE=srcf,STATUS='old')
OPEN(UNIT=50,FILE=otsf,STATUS='old')
OPEN(UNIT=60,FILE=otpf,STATUS='old')
READ(40,*)nr
READ(50,*)id1
READ(60,*)id1
DO i=1,nrp
   READ(40,*)id1,nslr(i)
   DO j=1,nslr(i)
      READ(40,*)idtto(i,j)
      READ(60,*)id1,id2,id3,id4,tto(i,j)
   ENDDO
ENDDO
spas=-1
nmp=0
DO i=nrp+1,nr
   READ(40,*)id1,nslr(i)
   DO j=1,nslr(i)
      READ(40,*)idtto(i,j)
      READ(50,*)id1,id2,id3,id4,tto(i,j)
      IF(idr(i).LE.nrp)THEN
         DO k=1,nslr(idr(i))
            IF(idtto(i,j).EQ.idtto(idr(i),k))THEN
               spas(i,j)=k
               nmp=nmp+1
               EXIT
            ENDIF
         ENDDO
      ENDIF
   ENDDO
ENDDO
CLOSE(40)
CLOSE(50)
CLOSE(60)
!
! Now compute the observed data and write to file
!
OPEN(UNIT=70,FILE=otf,STATUS='unknown')
rd1=0.0
DO i=nrp+1,nr
   DO j=1,nslr(i)
      IF(spas(i,j).GT.0)THEN
         id1=1
         tobs=tto(i,j)-vpvsr*tto(idr(i),spas(i,j))
         rd1=rd1+tobs**2
      ELSE
         id1=0
         tobs=-100.0
      ENDIF
      WRITE(70,*)id1,tobs
   ENDDO
ENDDO
CLOSE(70)
!
! Export total number of matching paths
!
OPEN(UNIT=110,FILE=sarf,STATUS='unknown')
WRITE(110,*)nmp
CLOSE(110)
!
! Now read in the ray paths, one at a time, and compute (1) Model data; (2) Frechet
! derivatives
!
OPEN(UNIT=80,FILE=mtf,STATUS='unknown')
OPEN(UNIT=90,FILE=frdf,FORM='unformatted',STATUS='unknown')
OPEN(UNIT=100,FILE=rayf,STATUS='old')
!
! Allocate memory to partial derivative array
!
ALLOCATE(fdm(0:nvz+1,0:nvx+1,0:nvr+1))
DO i=nrp+1,nr
   DO j=1,nslr(i)
      fdm=0.
      mdat=0.0
      iddat=0
      IF(spas(i,j).GT.0)THEN
         READ(100,*)
         READ(100,*)npt
         IF(npt.GT.1)THEN
            READ(100,*)rgro,rgxo,rgzo
            rgro=rgro-rearth
            rgxo=90.-rgxo*180./pi
            rgzo=rgzo*180./pi
            iddat=1
            plt=0.
            DO k=1,npt-1
               READ(100,*)rgr,rgx,rgz
               rgr=rgr-rearth
               rgx=90.-rgx*180./pi
               rgz=rgz*180./pi
!
!              Compute incremental path length
!
               rd1=(rearth+rgr)*COS(rgz*pi/180.)*SIN(rgx*pi/180.)
               rd1=rd1-(rearth+rgro)*COS(rgzo*pi/180.)*SIN(rgxo*pi/180.)
               rd2=(rearth+rgr)*SIN(rgz*pi/180.)*SIN(rgx*pi/180.)
               rd2=rd2-(rearth+rgro)*SIN(rgzo*pi/180.)*SIN(rgxo*pi/180.)
               rd3=(rearth+rgr)*COS(rgx*pi/180.)-(rearth+rgro)*COS(rgxo*pi/180.)
               dpl=SQRT(rd1**2+rd2**2+rd3**2)
               plt=plt+dpl
!
!              Locate B-spline cells in which ray segment
!              start point and end points lie in.
!     
               ivr=INT((gor-rgr)/dvr)+1
               ivx=INT((rgx-gox)/dvx)+1
               ivz=INT((rgz-goz)/dvz)+1
               ivro=INT((gor-rgro)/dvr)+1
               ivxo=INT((rgxo-gox)/dvx)+1
               ivzo=INT((rgzo-goz)/dvz)+1
!
!              Calculate up to three hit points between straight
!              ray segment and cell faces.
!
               nhp=0
               IF(ivr.NE.ivro)THEN
                  nhp=nhp+1
                  IF(ivr.GT.ivro)THEN
                     ri=gor-(ivr-1)*dvr
                  ELSE
                     ri=gor-ivr*dvr
                  ENDIF
                  IF(ABS(rgr-rgro).EQ.0.)THEN
                     vrat(nhp)=1.0e5
                  ELSE
                     vrat(nhp)=(ri-rgro)/(rgr-rgro)
                  ENDIF
                  chp(nhp)=1
               ENDIF
               IF(ivx.NE.ivxo)THEN
                  nhp=nhp+1
                  IF(ivx.GT.ivxo)THEN
                     xi=gox+(ivx-1)*dvx
                  ELSE
                     xi=gox+ivx*dvx
                  ENDIF
                  IF(ABS(rgx-rgxo).EQ.0.)THEN
                     rd1=1.0e5
                  ELSE
                     rd1=(xi-rgxo)/(rgx-rgxo)
                  ENDIF
                  IF(nhp.EQ.1)THEN
                     vrat(nhp)=rd1
                     chp(nhp)=2
                  ELSE
                     IF(rd1.GE.vrat(nhp-1))THEN
                        vrat(nhp)=rd1
                        chp(nhp)=2
                     ELSE
                        vrat(nhp)=vrat(nhp-1)
                        chp(nhp)=chp(nhp-1)
                        vrat(nhp-1)=rd1
                        chp(nhp-1)=2
                     ENDIF
                  ENDIF
               ENDIF
               IF(ivz.NE.ivzo)THEN
                  nhp=nhp+1
                  IF(ivz.GT.ivzo)THEN
                     zi=goz+(ivz-1)*dvz
                  ELSE
                     zi=goz+ivz*dvz
                  ENDIF
                  IF(ABS(rgz-rgzo).EQ.0.)THEN
                     rd1=1.0e5
                  ELSE
                     rd1=(zi-rgzo)/(rgz-rgzo)
                  ENDIF
                  IF(nhp.EQ.1)THEN
                     vrat(nhp)=rd1
                     chp(nhp)=3
                  ELSE IF(nhp.EQ.2)THEN
                     IF(rd1.GE.vrat(nhp-1))THEN
                        vrat(nhp)=rd1
                        chp(nhp)=3
                     ELSE
                        vrat(nhp)=vrat(nhp-1)
                        chp(nhp)=chp(nhp-1)
                        vrat(nhp-1)=rd1
                        chp(nhp-1)=3
                     ENDIF
                  ELSE
                     IF(rd1.GE.vrat(nhp-1))THEN
                        vrat(nhp)=rd1
                        chp(nhp)=3
                     ELSE IF(rd1.GE.vrat(nhp-2))THEN
                        vrat(nhp)=vrat(nhp-1)
                        chp(nhp)=chp(nhp-1)
                        vrat(nhp-1)=rd1
                        chp(nhp-1)=3
                     ELSE
                        vrat(nhp)=vrat(nhp-1)
                        chp(nhp)=chp(nhp-1)
                        vrat(nhp-1)=vrat(nhp-2)
                        chp(nhp-1)=chp(nhp-2)
                        vrat(nhp-2)=rd1
                        chp(nhp-2)=3
                     ENDIF
                  ENDIF
               ENDIF
               nhp=nhp+1
               vrat(nhp)=1.0
               chp(nhp)=0
!
!              Calculate u,v,w values of the first point
!
               drr=(gor-rgro)-(ivro-1)*dvr
               drx=(rgxo-gox)-(ivxo-1)*dvx
               drz=(rgzo-goz)-(ivzo-1)*dvz
               u=drr/dvr
               v=drx/dvx
               w=drz/dvz
!
!              Now compute Vp, d(Vp/Vs), Vs and dVs
!
               CALL bsplinevp(u,v,w,ivro,ivxo,ivzo,vel)
               CALL bsplinevpvs(u,v,w,ivro,ivxo,ivzo,velps)
               CALL bsplinevs(u,v,w,ivro,ivxo,ivzo,vels)
               CALL bsplinedvs(u,v,w,ivro,ivxo,ivzo,velds)
               ivrt=ivro
               ivxt=ivxo
               ivzt=ivzo               
!
!              Now loop through the one or more sub-segments of the
!              ray path segment and calculate partial derivatives
!
               DO ii=1,nhp
                  velo=vel
                  velpso=velps
                  velso=vels
                  veldso=velds
                  uio=ui
                  vio=vi
                  wio=wi
                  IF(ii.GT.1)THEN
                     IF(chp(ii-1).EQ.1)THEN
                        ivrt=ivr
                     ELSE IF(chp(ii-1).EQ.2)THEN
                        ivxt=ivx
                     ELSE IF(chp(ii-1).EQ.3)THEN
                        ivzt=ivz
                     ENDIF
                  ENDIF
                  rigz=rgzo+vrat(ii)*(rgz-rgzo)
                  rigx=rgxo+vrat(ii)*(rgx-rgxo)
                  rigr=rgro+vrat(ii)*(rgr-rgro)
!
!                 Calculate new u,v,w values
!
                  drr=(gor-rigr)-(ivrt-1)*dvr
                  drx=(rigx-gox)-(ivxt-1)*dvx
                  drz=(rigz-goz)-(ivzt-1)*dvz
                  u=drr/dvr
                  v=drx/dvx
                  w=drz/dvz
!
!                 Calculate updated velocity values
!
                  CALL bsplinevp(u,v,w,ivrt,ivxt,ivzt,vel)
                  CALL bsplinevpvs(u,v,w,ivrt,ivxt,ivzt,velps)
                  CALL bsplinevs(u,v,w,ivrt,ivxt,ivzt,vels)
                  CALL bsplinedvs(u,v,w,ivrt,ivxt,ivzt,velds)
!
!                 Calculate the incremental path length
!
                  IF(ii.EQ.1)THEN
                     dinc=vrat(ii)*dpl
                  ELSE
                     dinc=(vrat(ii)-vrat(ii-1))*dpl
                  ENDIF
!
!                 Now compute the 64 contributions to the partial
!                 derivatives.
!
                  DO l=1,4
                     DO m=1,4
                        DO n=1,4
                           rd1=(1.0+velds/vels)*ui(n)*vi(m)*wi(l)/vel
                           rd2=(1.0+veldso/velso)*uio(n)*vio(m)*wio(l)/velo
                           rd1=(rd1+rd2)*dinc/2.0
                           rd2=fdm(ivzt-2+l,ivxt-2+m,ivrt-2+n)
                           fdm(ivzt-2+l,ivxt-2+m,ivrt-2+n)=rd1+rd2
                        ENDDO
                     ENDDO
                  ENDDO
!
!                 Compute model data
!
                  mdat=mdat+((1.0+velds/vels)*velps/vel+(1.0+veldso/velso)*velpso/velo)*dinc/2.0
               ENDDO
               rgro=rgr
               rgxo=rgx
               rgzo=rgz
            ENDDO
         ELSE
            IF(npt.EQ.1)READ(100,*)
         ENDIF
      ELSE 
         READ(100,*)
         READ(100,*)npt
         IF(npt.GT.0)THEN
            DO k=1,npt
               READ(100,*)
            ENDDO
         ENDIF
      ENDIF
!
!     Write model data to output file
!
      WRITE(80,*)iddat,mdat
!
!     Write partial derivatives to output file
!
!
!     Determine the number of non-zero elements.
!
      isum=0
      DO ii=0,nvz+1
         DO k=0,nvx+1
            DO l=0,nvr+1
               IF(ABS(fdm(ii,k,l)).GE.ftol)isum=isum+1
            ENDDO
         ENDDO
      ENDDO
      WRITE(90)isum
      isum=0
      DO ii=0,nvz+1
         DO k=0,nvx+1
            DO l=0,nvr+1
               isum=isum+1
               IF(ABS(fdm(ii,k,l)).GE.ftol)WRITE(90)isum,fdm(ii,k,l)
            ENDDO
         ENDDO
      ENDDO
   ENDDO
ENDDO
CLOSE(80)
CLOSE(90)
CLOSE(100)
DEALLOCATE(dvpvs,vp,vs,dvs,fdm)
END PROGRAM frvpvs

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! SUBROUTINE BSPLINEVP
! Subroutine for computing P-velocity at a
! given point in the grid
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE bsplinevp(u,v,w,ivr,ivx,ivz,vel)
USE globalp
IMPLICIT NONE
INTEGER :: i,j,k,ivr,ivx,ivz
REAL :: u,v,w,vel,sumj,sumk
!
! vel = velocity at point
! ivr,ivx,ivz = Current grid cell location
! u,v,w = Proportional location in cell
! sumj,sumk = Real summation variables
!
ui(1)=(1.0-u)**3/6.0
ui(2)=(4.0-6.0*u**2+3.0*u**3)/6.0
ui(3)=(1.0+3.0*u+3.0*u**2-3.0*u**3)/6.0
ui(4)=u**3/6.0
vi(1)=(1.0-v)**3/6.0
vi(2)=(4.0-6.0*v**2+3.0*v**3)/6.0
vi(3)=(1.0+3.0*v+3.0*v**2-3.0*v**3)/6.0
vi(4)=v**3/6.0
wi(1)=(1.0-w)**3/6.0
wi(2)=(4.0-6.0*w**2+3.0*w**3)/6.0
wi(3)=(1.0+3.0*w+3.0*w**2-3.0*w**3)/6.0
wi(4)=w**3/6.0
vel=0.
DO i=1,4
   sumj=0.
   DO j=1,4
      sumk=0.
      DO k=1,4
         sumk=sumk+ui(k)*vp(ivr-2+k,ivx-2+j,ivz-2+i)
      ENDDO
      sumj=sumj+vi(j)*sumk
   ENDDO
   vel=vel+wi(i)*sumj
ENDDO
END SUBROUTINE bsplinevp

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! SUBROUTINE BSPLINEVPVS
! Subroutine for computing d(Vp/Vs) at a
! given point in the grid
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE bsplinevpvs(u,v,w,ivr,ivx,ivz,vel)
USE globalp
IMPLICIT NONE
INTEGER :: i,j,k,ivr,ivx,ivz
REAL :: u,v,w,vel,sumj,sumk
!
! vel = velocity at point
! ivr,ivx,ivz = Current grid cell location
! u,v,w = Proportional location in cell
! sumj,sumk = Real summation variables
!
vel=0.
DO i=1,4
   sumj=0.
   DO j=1,4
      sumk=0.
      DO k=1,4
         sumk=sumk+ui(k)*dvpvs(ivr-2+k,ivx-2+j,ivz-2+i)
      ENDDO
      sumj=sumj+vi(j)*sumk
   ENDDO
   vel=vel+wi(i)*sumj
ENDDO
END SUBROUTINE bsplinevpvs

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! SUBROUTINE BSPLINEVS
! Subroutine for computing Vs at a
! given point in the grid
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE bsplinevs(u,v,w,ivr,ivx,ivz,vel)
USE globalp
IMPLICIT NONE
INTEGER :: i,j,k,ivr,ivx,ivz
REAL :: u,v,w,vel,sumj,sumk
!
! vel = velocity at point
! ivr,ivx,ivz = Current grid cell location
! u,v,w = Proportional location in cell
! sumj,sumk = Real summation variables
!
vel=0.
DO i=1,4
   sumj=0.
   DO j=1,4
      sumk=0.
      DO k=1,4
         sumk=sumk+ui(k)*vs(ivr-2+k,ivx-2+j,ivz-2+i)
      ENDDO
      sumj=sumj+vi(j)*sumk
   ENDDO
   vel=vel+wi(i)*sumj
ENDDO
END SUBROUTINE bsplinevs

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! SUBROUTINE BSPLINEDVS
! Subroutine for computing d(Vs) at a
! given point in the grid
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE bsplinedvs(u,v,w,ivr,ivx,ivz,vel)
USE globalp
IMPLICIT NONE
INTEGER :: i,j,k,ivr,ivx,ivz
REAL :: u,v,w,vel,sumj,sumk
!
! vel = velocity at point
! ivr,ivx,ivz = Current grid cell location
! u,v,w = Proportional location in cell
! sumj,sumk = Real summation variables
!
vel=0.
DO i=1,4
   sumj=0.
   DO j=1,4
      sumk=0.
      DO k=1,4
         sumk=sumk+ui(k)*dvs(ivr-2+k,ivx-2+j,ivz-2+i)
      ENDDO
      sumj=sumj+vi(j)*sumk
   ENDDO
   vel=vel+wi(i)*sumj
ENDDO
END SUBROUTINE bsplinedvs
