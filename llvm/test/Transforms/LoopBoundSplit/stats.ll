; REQUIRES: asserts
; RUN: opt -passes=loop-bound-split -stats -disable-output < %s 2>&1 | FileCheck %s

; CHECK: 1 loop-bound-split - Number of loops whose bounds were split

define void @constant_split_loop_bound_and_exit_cond_inc_with_sgt(ptr noalias %src, ptr noalias %dst, i64 %n) {
loop.ph:
  br label %loop

loop:
  %iv = phi i64 [ %inc, %for.inc ], [ 0, %loop.ph ]
  %cmp = icmp ult i64 %iv, 10
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %src.arrayidx = getelementptr inbounds i64, ptr %src, i64 %iv
  %val = load i64, ptr %src.arrayidx
  %dst.arrayidx = getelementptr inbounds i64, ptr %dst, i64 %iv
  store i64 %val, ptr %dst.arrayidx
  br label %for.inc

if.else:
  br label %for.inc

for.inc:
  %inc = add nuw nsw i64 %iv, 1
  %cond = icmp sgt i64 %inc, %n
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}
