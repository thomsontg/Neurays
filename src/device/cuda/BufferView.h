//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include <src/sutil/Preprocessor.h>

template <typename T>
struct BufferView
{
    CUdeviceptr    data           CONST_STATIC_INIT( 0 );
    unsigned int   count          CONST_STATIC_INIT( 0 );
    unsigned short byte_stride    CONST_STATIC_INIT( 0 );
    unsigned short elmt_byte_size CONST_STATIC_INIT( 0 );

    SUTIL_HOSTDEVICE bool isValid() const
    { return static_cast<bool>( data ); }

    SUTIL_HOSTDEVICE operator bool() const
    { return isValid(); }

    SUTIL_HOSTDEVICE const T& operator[]( unsigned int idx ) const
    {
        return *reinterpret_cast<T*>( data + idx*(byte_stride ? byte_stride : sizeof( T ) ) ); 
    }
    
    SUTIL_HOSTDEVICE void append(int val)
    {
        count += 1;
        *reinterpret_cast<T*>(data + (count) * sizeof(T)) = val;
    }
};

typedef BufferView<unsigned int> GenericBufferView;

